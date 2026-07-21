"""
Central configuration. Everything tunable lives here so the report's
hyperparameter table matches the code exactly. Import CONFIG everywhere.
"""

CONFIG = {
    # ---- retrieval ----
    "embed_model": "BAAI/bge-large-en-v1.5",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L12-v2",
    "bm25_topk": 20,          # top-k from sparse retriever
    "dense_topk": 20,         # top-k from dense retriever
    "rrf_k": 60,              # RRF smoothing constant (standard value)
    "rerank_topk": 5,         # docs kept after reranking as evidence

    # ---- BM25 params (standard Okapi values, reported in your paper) ----
    "bm25_k1": 1.5,
    "bm25_b": 0.75,

    # ---- confidence estimation weights ----
    # Conf = alpha*token + beta*consistency + gamma*Var(R)
    # NOTE: gamma is now a REWARD (sign flipped from penalty) per signal diagnostics:
    # high rerank-score variance => reranker cleanly separated relevant/irrelevant => good.
    # beta kept small: S2 (evidence consistency) found non-predictive on short-answer QA
    # (retained in framework for completeness; reported in ablation).
    "alpha": 0.7,             # token-probability / self-consistency weight (primary signal)
    "beta": 0.05,             # evidence-consistency weight (near-zero; non-predictive)
    "gamma": 0.25,            # retrieval-variance REWARD weight (now positive contribution)

    # ---- adaptive policy ----
    "tau": 0.6,               # confidence threshold to STOP
    "T_max": 3,               # retrieval budget (max iterations)
    "k_step": 5,              # extra docs pulled in per RETRIEVE_MORE round

    # ---- generation ----
    "local_model": "Qwen/Qwen2.5-1.5B-Instruct",  # fits 4GB VRAM in fp16, exposes token logprobs
    "claude_model": "claude-haiku-4-5-20251001",   # secondary backend (no logprobs -> self-consistency)
    "ollama_model": "llama3.2:3b",                  # open-source tier via Ollama (self-consistency)
    "n_consistency_samples": 3,    # for Claude/Ollama paths: samples to estimate self-consistency confidence
    "max_new_tokens": 64,     # short-answer QA needs very little
    "gen_temperature": 0.0,   # deterministic answers for reproducibility

    # ---- eval ----
    "seed": 42,
}
