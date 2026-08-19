# -*- coding: utf-8 -*-
"""Gene Ontology enrichment of the SPEF1 co-expression module (manuscript Fig. 7).

Draws the dataset GO enrichment vertically, in the same style as the STRING
enrichment figure: one panel per GO category (biological process, cellular
component, molecular function), dot size = gene count, colour = -log10 FDR.
Reads ``GO_coexpressed.csv`` produced by ``SPEF1_analysis/main.py``.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results/figures"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved:", name)


def go_string_style():
    go = pd.read_csv(ROOT / "results/tables/GO_coexpressed.csv")
    cats = ["Biological process", "Cellular component", "Molecular function"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 13))
    cmap = plt.cm.viridis_r
    for ax, cat, letter in zip(axes, cats, "ABC"):
        sub = go[go.category == cat].sort_values("q_value_BH").head(10).iloc[::-1]
        x = -np.log10(sub["q_value_BH"].clip(lower=1e-300))
        sizes = 40 + sub["n_overlap"] / sub["n_overlap"].max() * 320
        colors = cmap((x - x.min()) / (x.max() - x.min() + 1e-9))
        for yi, (xi, term) in enumerate(zip(x, sub["go_term"])):
            ax.plot([0, xi], [yi, yi], color="0.7", lw=1.4, zorder=1)
        ax.scatter(x, range(len(sub)), s=sizes, c=colors, edgecolor="0.3", linewidth=0.5, zorder=2)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([t if len(t) <= 44 else t[:41] + "…" for t in sub["go_term"]], fontsize=10)
        ax.set_xlabel("−log10 FDR"); ax.set_xlim(0, x.max() * 1.12)
        ax.set_title(f"{letter}   {cat}", loc="left", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        # size legend (gene count)
        for gc in [sub["n_overlap"].min(), sub["n_overlap"].median(), sub["n_overlap"].max()]:
            ax.scatter([], [], s=40 + gc / sub["n_overlap"].max() * 320, c="0.6",
                       edgecolor="0.3", label=f"{int(gc)}")
        ax.legend(title="Genes", fontsize=8, title_fontsize=9, loc="lower right", frameon=False)
    fig.suptitle("Gene Ontology enrichment of the SPEF1 co-expression module (GSE101108)", y=1.005, fontsize=13)
    fig.tight_layout()
    save(fig, "Figure_SPEF1_dataset_GO_string_style")


if __name__ == "__main__":
    go_string_style()
