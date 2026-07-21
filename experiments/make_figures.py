"""
Generate the 5 data-driven figures for the AB-RAG report.
All numbers hard-coded from the verified result JSONs (cited in REPORT_TABLES.md).
Output: high-res PNGs (300 dpi) in figures/
Style: clean academic — muted palette, Times-like serif, clear labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ---- global style: professional RAG-paper look ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "legend.fontsize": 9.5,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "figure.dpi": 300,
})

# muted academic palette
C = {
    "bm25":   "#8C8C8C",
    "dense":  "#2E6F9E",
    "hybrid": "#E0A458",
    "rerank": "#5B8C5A",
    "static": "#A6A6A6",
    "abrag":  "#2E6F9E",
    "high":   "#5B8C5A",
    "low":    "#C25B56",
    "qwen":   "#8C8C8C",
    "llama":  "#5B8C5A",
    "claude": "#2E6F9E",
}

def save(fig, name):
    path = f"figures/{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)

# =====================================================================
# Figure 5.1 — Open-retrieval recall bars (HotpotQA + TriviaQA side by side)
# =====================================================================
ks = ["@5", "@10", "@20", "@50"]
hotpot = {
    "BM25":   [58.7, 72.8, 80.9, 86.2],
    "Dense":  [89.2, 95.3, 97.2, 98.3],
    "Hybrid": [79.8, 93.3, 96.9, 97.6],
    "Reranked":[86.6, 94.0, 97.3, 97.6],
}
trivia = {
    "BM25":   [43.6, 54.1, 63.7, 69.6],
    "Dense":  [66.9, 80.6, 89.1, 95.3],
    "Hybrid": [59.4, 75.8, 86.2, 90.8],
    "Reranked":[67.0, 80.3, 88.4, 90.8],
}
cols = [C["bm25"], C["dense"], C["hybrid"], C["rerank"]]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, data, title, corpus in [
    (axes[0], hotpot, "HotpotQA (n=500, corpus=4,963)", ""),
    (axes[1], trivia, "TriviaQA (n=200, corpus=2,582)", ""),
]:
    x = np.arange(len(ks)); w = 0.2
    for i, (m, vals) in enumerate(data.items()):
        ax.bar(x + (i-1.5)*w, vals, w, label=m, color=cols[i], edgecolor="white", linewidth=0.5)
    ax.set_title(title); ax.set_xlabel("Recall@k"); ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_ylim(0, 105); ax.set_axisbelow(True)
axes[0].set_ylabel("Recall (%)")
axes[1].legend(loc="lower right", ncol=2)
fig.suptitle("Open-Retrieval Recall by Method", y=1.02, fontsize=13, fontweight="bold")
save(fig, "fig5_1_recall_bars")

# =====================================================================
# Figure 5.2 — EM: Static vs AB-RAG across backbones
# =====================================================================
labels = ["Qwen-1.5B\nHotpotQA", "Llama-3.2-3B\nHotpotQA", "Claude\nHotpotQA", "Claude\nTriviaQA"]
static_em = [34.5, 39.5, 32.5, 35.5]
abrag_em  = [34.5, 45.0, 33.0, 40.0]
# 95% CI half-widths (from bootstrap, approx symmetric) for AB-RAG
abrag_ci = [(41.0-28.5)/2, (52.0-38.5)/2, (39.5-26.5)/2, (47.0-33.5)/2]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.bar(x - w/2, static_em, w, label="Static RAG", color=C["static"], edgecolor="white")
ax.bar(x + w/2, abrag_em, w, yerr=abrag_ci, capsize=4, label="AB-RAG",
       color=C["abrag"], edgecolor="white", error_kw={"elinewidth":1, "ecolor":"#333"})
for i,(s,a) in enumerate(zip(static_em, abrag_em)):
    ax.text(x[i]-w/2, s+0.6, f"{s}", ha="center", fontsize=8.5)
    ax.text(x[i]+w/2, a+abrag_ci[i]+0.8, f"{a}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Exact Match (%)")
ax.set_title("Static RAG vs AB-RAG (Exact Match, 95% CI on AB-RAG)")
ax.set_ylim(0, 58); ax.legend(loc="upper left"); ax.set_axisbelow(True)
save(fig, "fig5_2_em_static_vs_abrag")

# =====================================================================
# Figure 5.3 — Confidence separation (high vs low conf EM)
# =====================================================================
labels = ["Qwen-1.5B\nHotpotQA", "Llama-3.2-3B\nHotpotQA", "Claude\nHotpotQA", "Claude\nTriviaQA"]
high = [34.7, 51.6, 36.5, 57.6]
low  = [0.0, 22.2, 0.0, 0.0]
high_n = [199, 155, 181, 139]; low_n = [1, 45, 19, 61]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(8.4, 4.6))
b1 = ax.bar(x - w/2, high, w, label="High-confidence (≥ τ)", color=C["high"], edgecolor="white")
b2 = ax.bar(x + w/2, low, w, label="Low-confidence (< τ)", color=C["low"], edgecolor="white")
for i in range(len(labels)):
    ax.text(x[i]-w/2, high[i]+0.8, f"{high[i]}%\n(n={high_n[i]})", ha="center", fontsize=7.8)
    ax.text(x[i]+w/2, low[i]+0.8, f"{low[i]}%\n(n={low_n[i]})", ha="center", fontsize=7.8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Exact Match (%)")
ax.set_title("Confidence Separates Correct from Incorrect Answers (τ = 0.6)")
ax.set_ylim(0, 70); ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12)); ax.set_axisbelow(True)
ax.annotate("Qwen low-conf n=1\n(not interpretable)", xy=(0+w/2, 2), xytext=(0.30, 16),
            fontsize=7, color="#888", ha="left",
            arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.7))
save(fig, "fig5_3_confidence_separation")

# =====================================================================
# Figure 5.4 — Threshold sweep (cost-accuracy tradeoff)
# =====================================================================
# Claude HotpotQA sweep (from RESEARCH_LOG §2.5) and TriviaQA (§8), Qwen flat (§2.4)
tau_c = [0.45, 0.50, 0.55]
em_c  = [31.0, 31.5, 33.0]; it_c = [1.00, 1.15, 1.36]
tau_t = [0.45, 0.50, 0.55, 0.60]
em_t  = [40.0, 40.0, 40.0, 40.0]; it_t = [1.33, 1.33, 1.78, 1.81]
# Qwen flat
tau_q = [0.45, 0.55, 0.65, 0.80]; em_q = [35.5, 35.0, 34.7, 34.5]; it_q=[1.00,1.05,1.08,1.09]
fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.plot(it_c, em_c, "o-", color=C["claude"], label="Claude HotpotQA", linewidth=1.8, markersize=6)
ax.plot(it_t, em_t, "s-", color=C["llama"], label="Claude TriviaQA", linewidth=1.8, markersize=6)
ax.plot(it_q, em_q, "^--", color=C["qwen"], label="Qwen-1.5B HotpotQA (flat)", linewidth=1.5, markersize=6)
# annotate tau at endpoints
ax.annotate("τ=0.55", (it_c[-1], em_c[-1]), textcoords="offset points", xytext=(6,-2), fontsize=8, color=C["claude"])
ax.annotate("τ=0.45", (it_c[0], em_c[0]), textcoords="offset points", xytext=(6,-10), fontsize=8, color=C["claude"])
ax.set_xlabel("Average retrieval iterations per query (cost)")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Cost–Accuracy Tradeoff via Confidence Threshold τ")
ax.legend(loc="center right"); ax.set_axisbelow(True)
save(fig, "fig5_4_sweep_tradeoff")

# =====================================================================
# Figure 5.5 — Single-signal AUROC ablation across backbones
# =====================================================================
backs = ["Qwen\nHotpotQA", "Llama\nHotpotQA", "Claude\nHotpotQA", "Claude\nTriviaQA"]
c_tok = [0.607, 0.651, 0.769, 0.776]
c_con = [0.517, 0.568, 0.353, 0.374]
v_ret = [0.429, 0.550, 0.572, 0.594]  # Claude-Trivia uses diagnostic +var value
x = np.arange(len(backs)); w = 0.26
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(x - w, c_tok, w, label="S1: token / self-consistency", color=C["claude"], edgecolor="white")
ax.bar(x,     c_con, w, label="S2: evidence consistency", color=C["low"], edgecolor="white")
ax.bar(x + w, v_ret, w, label="S3: retrieval variance (reward)", color=C["rerank"], edgecolor="white")
ax.axhline(0.5, color="#888", linestyle="--", linewidth=1)
ax.text(len(backs)-0.5, 0.508, "0.5 (chance)", fontsize=8, color="#888", ha="right")
ax.set_xticks(x); ax.set_xticklabels(backs); ax.set_ylabel("AUROC (confidence vs correctness)")
ax.set_title("Single-Signal Predictiveness Across Backbones")
ax.set_ylim(0.30, 0.82); ax.legend(loc="upper left", ncol=1); ax.set_axisbelow(True)
save(fig, "fig5_5_signal_ablation")

print("\nAll 5 data figures generated.")
