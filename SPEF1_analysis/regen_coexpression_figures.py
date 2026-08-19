# -*- coding: utf-8 -*-
"""Clustering heatmap + correlation matrix (SPEF1 + 20 genes), HGSC/LGSC split."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT.parent / "GSE101108_pipeline"
FIG = ROOT / "results/figures"

# --- data: counts, fine histotype (HGSC/LGSC), normalization ---
counts = pd.read_csv(PIPE / "data/processed/GSE101108_complete_expression_matrix.csv.gz", index_col=0)
counts.columns = [c.split("|")[0] for c in counts.columns]
hist = pd.read_excel(PIPE / "data/metadata/GSE101108_histotype_classification.xlsx", sheet_name="classificazione")
FINE = {"HGSC": "HGSC", "LGSC": "LGSC", "CCC": "Clear cell", "EC": "Endometrioid", "MC": "Mucinous"}
hist["fine"] = hist["histotype_original"].map(FINE)
hist = hist.dropna(subset=["fine"]).set_index("gsm_id")
samples = [c for c in counts.columns if c in hist.index]
counts = counts[samples]
fine = hist.loc[samples, "fine"]

# size factors median-of-ratios
m = counts.to_numpy(float); usable = (m > 0).all(1)
sf = np.exp(np.median(np.log(m[usable]) - np.log(m[usable]).mean(1, keepdims=True), 0))
logx = np.log2(counts / sf + 1.0)

# genes: SPEF1 + top 20 co-expressed (actual symbols)
co = pd.read_csv(ROOT / "results/tables/SPEF1_coexpression_all.csv.gz", index_col=0)
top = [g for g in co.sort_values("rho", ascending=False).index if not str(g).startswith("ENSG")][:20]
genes = ["SPEF1"] + [g for g in top if g in logx.index]
sub = logx.loc[genes]

ORDER = ["HGSC", "LGSC", "Clear cell", "Endometrioid", "Mucinous"]
HCOL = {"HGSC": "#4C78A8", "LGSC": "#72B7B2", "Clear cell": "#F58518",
        "Endometrioid": "#54A24B", "Mucinous": "#E45756"}

# ================= 1. HEATMAP (clustermap) =================
z = sub.sub(sub.mean(1), axis=0).div(sub.std(1).replace(0, np.nan), axis=0)
col_colors = fine.map(HCOL)
g = sns.clustermap(
    z, cmap="RdBu_r", vmin=-2, vmax=2, col_colors=col_colors,
    col_cluster=True, row_cluster=True, xticklabels=False, yticklabels=True,
    figsize=(13, 8), dendrogram_ratio=(0.08, 0.12), cbar_pos=(0.02, 0.83, 0.02, 0.13),
    colors_ratio=0.03, linewidths=0)
g.ax_heatmap.set_xlabel(f"Ovarian carcinomas (n = {len(samples)}), clustered")
g.ax_heatmap.set_ylabel("")
g.ax_cbar.set_title("z-score", fontsize=9)
from matplotlib.patches import Patch
g.ax_heatmap.legend(handles=[Patch(facecolor=HCOL[h], label=f"{h} (n={int((fine==h).sum())})") for h in ORDER],
                    title="Histotype", bbox_to_anchor=(1.06, 1.0), loc="upper left",
                    fontsize=9, title_fontsize=10, frameon=False)
g.fig.suptitle("Unsupervised clustering of SPEF1 and its co-expression module across histotypes (GSE101108)",
               y=1.01, fontsize=12)
g.savefig(FIG / "Figure_SPEF1_heatmap_clustering.png", dpi=300, bbox_inches="tight")
g.savefig(FIG / "Figure_SPEF1_heatmap_clustering.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close("all")
print("heatmap saved")

# ================= 2. CORRELATION MATRIX (with significance stars) =================
n = len(genes)
R = np.ones((n, n)); P = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            rho, p = stats.spearmanr(sub.iloc[i], sub.iloc[j])
            R[i, j] = rho; P[i, j] = p
Rdf = pd.DataFrame(R, index=genes, columns=genes)
def star(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
annot = np.empty((n, n), dtype=object)
for i in range(n):
    for j in range(n):
        annot[i, j] = f"{R[i,j]:.2f}\n{star(P[i,j])}" if i != j else "1"
mask = np.triu(np.ones((n, n), bool), k=1)
fig, ax = plt.subplots(figsize=(13.5, 12))
sns.heatmap(Rdf, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, annot=annot, fmt="",
            annot_kws={"fontsize": 7}, square=True, linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.5, "label": "Spearman ρ"}, ax=ax)
ax.set_title("Correlation matrix of SPEF1 and its co-expressed genes in GSE101108\n"
             "(Spearman ρ; ***p<0.001, **p<0.01, *p<0.05)", fontsize=12, loc="left")
plt.xticks(rotation=90, fontsize=9); plt.yticks(rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "Figure_SPEF1_correlation_matrix.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "Figure_SPEF1_correlation_matrix.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("correlation matrix saved")
print("genes used:", genes)
print("histotypes (dataset):", fine.value_counts().to_dict())
