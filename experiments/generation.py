"""
generation.py
-------------
Backend-agnostic answer generation. Produces, for each query+evidence:
  - answer text
  - a token-probability confidence signal (Signal 1)

LOCAL backend (Qwen2.5-1.5B-Instruct):
  Uses HuggingFace generate() with output_scores=True to get REAL per-token
  log probabilities. Signal 1 = mean token probability, exactly as designed.

CLAUDE backend (claude API, no logprobs available):
  Signal 1 is approximated by SELF-CONSISTENCY: sample the answer k times at
  temperature>0 and measure agreement (fraction of samples whose normalized
  answer equals the modal answer). High agreement -> high confidence.
  This is the honest substitute for logprobs on a closed API.

Both backends share the SAME prompt template so the comparison is fair.
"""
import os
import numpy as np
from collections import Counter

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the evidence below. "
    "Give a short, direct answer (a few words). If the evidence is insufficient, say 'unknown'.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Question: {question}\nAnswer:"
)


def build_prompt(question, evidence_texts):
    ev = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts))
    return PROMPT_TEMPLATE.format(evidence=ev, question=question)


# ----------------------------------------------------------------------
# LOCAL BACKEND
# ----------------------------------------------------------------------
class LocalGenerator:
    def __init__(self, model_name, device="cuda", max_new_tokens=64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        print(f"loading local model {model_name} on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )
        self.model.eval()

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                 # deterministic
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        # decode only the newly generated tokens
        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        # --- Signal 1: real per-token probabilities ---
        token_logprobs = []
        for i, score in enumerate(out.scores):           # one score tensor per step
            logprobs = self.torch.log_softmax(score[0], dim=-1)
            tok_id = gen_ids[i]
            token_logprobs.append(float(logprobs[tok_id].item()))
        return {
            "answer": answer,
            "token_logprobs": token_logprobs,    # natural log probs
            "signal1_kind": "token_logprob",
        }


# ----------------------------------------------------------------------
# CLAUDE BACKEND
# ----------------------------------------------------------------------
class ClaudeGenerator:
    def __init__(self, model_name, n_samples=3, max_tokens=64):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model_name
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set. Run: "
                               "$env:ANTHROPIC_API_KEY = 'sk-ant-...'")

    @staticmethod
    def _norm(s):
        return " ".join(s.lower().strip().split())

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        samples = []
        for _ in range(self.n_samples):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.7,                 # >0 so samples can differ
                messages=[{"role": "user", "content": prompt}],
            )
            txt = "".join(b.text for b in msg.content if b.type == "text").strip()
            samples.append(txt)
        # modal answer + self-consistency confidence
        norms = [self._norm(s) for s in samples]
        counts = Counter(norms)
        modal_norm, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(samples)         # in [0,1]
        # pick a representative raw answer matching the modal normalized form
        answer = next(s for s in samples if self._norm(s) == modal_norm)
        return {
            "answer": answer,
            "self_consistency": consistency,    # used as Signal 1 proxy
            "samples": samples,
            "signal1_kind": "self_consistency",
        }


def get_generator(backend, config, device="cuda"):
    if backend == "local":
        return LocalGenerator(config["local_model"], device=device,
                              max_new_tokens=config["max_new_tokens"])
    elif backend == "claude":
        return ClaudeGenerator(config["claude_model"],
                               n_samples=config["n_consistency_samples"],
                               max_tokens=config["max_new_tokens"])
    elif backend == "ollama":
        return OllamaGenerator(config["ollama_model"],
                               n_samples=config["n_consistency_samples"],
                               max_tokens=config["max_new_tokens"])
    raise ValueError(f"unknown backend: {backend}")


# ----------------------------------------------------------------------
# OLLAMA BACKEND (open-source local model, no logprobs -> self-consistency)
# ----------------------------------------------------------------------
class OllamaGenerator:
    def __init__(self, model_name, n_samples=3, max_tokens=64):
        import ollama
        self.ollama = ollama
        self.model = model_name
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        # fail early with a clear message if model isn't pulled
        try:
            ollama.show(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Ollama model '{model_name}' not found. Run:  ollama pull {model_name}\n"
                f"(and make sure the Ollama app/service is running)\nDetail: {str(e)[:120]}")

    @staticmethod
    def _norm(s):
        return " ".join(s.lower().strip().split())

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        samples = []
        for _ in range(self.n_samples):
            resp = self.ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": self.max_tokens},
            )
            samples.append(resp["response"].strip())
        norms = [self._norm(s) for s in samples]
        counts = Counter(norms)
        modal_norm, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(samples)
        answer = next(s for s in samples if self._norm(s) == modal_norm)
        return {
            "answer": answer,
            "self_consistency": consistency,
            "samples": samples,
            "signal1_kind": "self_consistency",
        }
