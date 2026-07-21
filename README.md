# AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation

Training-free, backbone-agnostic RAG that decides *how much* to retrieve per
question - generate, estimate confidence from three signals, retrieve more only
if unsure, stop when confident or when the budget runs out.

📄 **Paper:** [arXiv:2606.29090](https://arxiv.org/abs/2606.29090) (cs.CL, cross-listed cs.IR / cs.AI) · journal version in submission

## The idea

Standard RAG retrieves a fixed number of passages for every query — over-retrieving
for easy questions (wasting compute, context, and API money) and under-retrieving
for hard ones. AB-RAG makes retrieval depth adaptive and *budgeted*:

```
t = 1, K = rerank_topk
loop:
    evidence = top-K reranked passages          (BM25 ∪ dense → RRF → cross-encoder)
    answer   = LLM(question, evidence)
    conf     = α·S1 + β·S2 + γ·S3
    if conf ≥ τ:      STOP  (confident)
    elif t ≥ T_max:   STOP  (budget exhausted)
    else:             K += k_step; t += 1       (retrieve more)
```

The three confidence signals, all training-free:

- **S1 — model certainty.** Mean token probability where logprobs are exposed
  (local HF models); self-consistency sampling as the honest substitute on
  closed APIs (Claude).
- **S2 — evidence–answer consistency.** Cosine similarity between the answer
  embedding and the mean evidence embedding.
- **S3 — retrieval score variance.** Variance of the reranker scores over the
  evidence set. Originally a penalty; signal diagnostics
  (`src/07_signal_diagnostics.py`) showed the sign was backwards, and it is
  used as corrected.

Evaluated on three backbone classes across all three common serving paths:
Qwen2.5-1.5B-Instruct (local HuggingFace, real logprobs), Llama-3.2-3B
(Ollama), and Claude Haiku (API, self-consistency).

## Headline results

| Finding | Result |
|---|---|
| Confidence separation, TriviaQA/Claude | **57.6% EM** (high-conf) vs **0% EM** (low-conf) |
| Confidence separation, HotpotQA/Claude | 36.5% vs 0.0% EM (τ = 0.6) |
| Llama-3.2-3B, adaptive vs static | **+5.5 EM** |
| Claude Haiku, TriviaQA, adaptive vs static | **+4.5 EM** |
| Qwen-1.5B | null result — no significant gain, reported as-is |
| S2 for short factoid answers | near-zero useful weight — documented failure |
| S3 sign | anti-predictive as designed; diagnosed and corrected via AUROC |

Negative and null findings are reported deliberately — see §Discussion in the
paper. The entire study ran on one consumer laptop with a few dollars of API
spend.

## Repository layout

```
src/                    pipeline, numbered in run order
  00_prepare_data.py      HotpotQA (distractor) + NQ → one uniform JSONL schema
  00b_prepare_triviaqa.py TriviaQA prep (streaming, no 17 GB download)
  01_retrieval.py         per-question hybrid retrieval: BM25 + FAISS dense, RRF fusion
  01b_open_retrieval.py   pooled open-retrieval corpus (the harder, non-saturating setting)
  02_rerank.py            cross-encoder reranking on the candidate pool
  03_generate_and_confidence.py  answers + all three confidence signals per question
  04_adaptive_loop.py     THE AB-RAG ALGORITHM — budgeted adaptive loop w/ full traces
  05_experiments.py       static vs adaptive headline comparison, EM/F1/AUROC
  06_ablations_and_sweep.py  τ sweep + signal ablations, replayed from traces (no LLM re-runs)
  07_signal_diagnostics.py   why S2/S3 underperform; alternative formulations, AUROC each
  confidence.py           the three signals + weighted combination (pure numpy)
  generation.py           local HF / Ollama / Claude backends, shared prompt
  make_figures.py         the 5 data-driven figures (300 dpi)
configs/config.py       every hyperparameter in one dict
results/                committed JSON/JSONL outputs backing the paper's tables
figures/                generated plots
data/                   cached datasets (gitignored — rebuilt by 00_*.py)
paper/                  arXiv LaTeX source + compiled PDF
```

## Setup

```bash
git clone https://github.com/Killer1GhosT/adaptive-budgeted-rag.git
cd adaptive-budgeted-rag
pip install -r requirements.txt
```

For the Claude backend, export `ANTHROPIC_API_KEY`. For the Ollama backend,
install [Ollama](https://ollama.com) and `ollama pull` the model named in
`configs/config.py`.

## Reproducing the paper

Run in numeric order — each stage writes to `results/` and later stages reuse
those files, so the expensive LLM stages run once:

```bash
python src/00_prepare_data.py
python src/00b_prepare_triviaqa.py
python src/01_retrieval.py  --dataset hotpotqa --n 500
python src/01b_open_retrieval.py --dataset hotpotqa --n 500
python src/02_rerank.py     --dataset hotpotqa --n 200
python src/03_generate_and_confidence.py --dataset hotpotqa --n 200 --backend local
python src/04_adaptive_loop.py --dataset hotpotqa --n 200 --backend local
python src/05_experiments.py   --dataset hotpotqa --n 200 --backend local
python src/06_ablations_and_sweep.py --dataset hotpotqa --backend local
python src/07_signal_diagnostics.py  --dataset hotpotqa --backend claude --n 200
python src/make_figures.py
```

Swap `--backend local` for `claude` or `ollama` to reproduce the other
backbones. The committed `results/` files are the exact outputs behind the
paper's tables, so every number is checkable without re-running anything.

## Citation

```bibtex
@article{kamthan2026abrag,
  title   = {AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering},
  author  = {Kamthan, Ansh},
  year    = {2026},
  eprint  = {2606.29090},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License

Code: MIT (see `LICENSE`). Paper text and figures via arXiv under its standard
license; the journal version is subject to the publisher's terms once accepted.

## Contact

Ansh Kamthan — ak14164@nyu.edu
