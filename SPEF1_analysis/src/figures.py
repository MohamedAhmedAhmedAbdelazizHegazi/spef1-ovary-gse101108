"""Figures for the manuscript (PNG and TIFF at 300 dpi).

Restrained, print-ready style: no heavy grid, readable fonts, palette
consistent across the figures.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: Per-histotype palette (consistent across all figures).
HISTOTYPE_COLORS = {
    "Serous": "#4C78A8",
    "Clear cell": "#F58518",
    "Endometrioid": "#54A24B",
    "Mucinous": "#E45756",
}
NEUTRAL = "#6E7B8B"
ACCENT = "#B279A2"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _save(fig: plt.Figure, path: Path, dpi: int, also_tiff: bool = True) -> Path | None:
    """Save the figure as PNG (and TIFF for submission)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if also_tiff:
            fig.savefig(path.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight",
                        pil_kwargs={"compression": "tiff_lzw"})
        LOGGER.info("Figura salvata: %s", path.name)
        return path
    except (OSError, PermissionError) as exc:
        LOGGER.error("Impossibile salvare %s: %s", path, exc)
        return None
    finally:
        plt.close(fig)


def _stars(qvalue: float) -> str:
    """Star notation for the corrected significance."""
    if qvalue is None or np.isnan(qvalue):
        return "n/d"
    if qvalue < 0.001:
        return "***"
    if qvalue < 0.01:
        return "**"
    if qvalue < 0.05:
        return "*"
    return "ns"


def _annotate_pairs(
    ax: plt.Axes,
    pairs: pd.DataFrame,
    order: Sequence[str],
    top: float,
    only_significant: bool = True,
    step: float | None = None,
) -> None:
    """Draw the significant comparison bars above the plot."""
    if pairs.empty:
        return
    positions = {label: i + 1 for i, label in enumerate(order)}
    selected = pairs[pairs["q_value_BH"] < 0.05] if only_significant else pairs
    if selected.empty:
        return
    step = step if step is not None else max(top * 0.09, 0.25)
    level = top + step
    for _, row in selected.iterrows():
        x1 = positions.get(row["group_1"])
        x2 = positions.get(row["group_2"])
        if x1 is None or x2 is None:
            continue
        ax.plot([x1, x1, x2, x2], [level, level + step * 0.25, level + step * 0.25, level],
                color="black", linewidth=0.8)
        ax.text((x1 + x2) / 2, level + step * 0.3, _stars(row["q_value_BH"]),
                ha="center", va="bottom", fontsize=9)
        level += step * 1.1


# --------------------------------------------------------------------------- #
# Figure 1: expression and detectability per histotype
# --------------------------------------------------------------------------- #


def plot_target_by_histotype(
    per_sample: pd.DataFrame,
    detection: pd.DataFrame,
    expression_pairs: pd.DataFrame,
    detection_pairs: pd.DataFrame,
    kruskal: Mapping[str, float],
    order: Sequence[str],
    target: str,
    path: Path,
    dpi: int = 300,
) -> Path | None:
    """Panel A: levels per histotype. Panel B: detection rate."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax = axes[0]
    data, labels, colors = [], [], []
    for label in order:
        values = per_sample.loc[per_sample["histotype"] == label, "log2_normalized"]
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(f"{label}\n(n={values.size})")
        colors.append(HISTOTYPE_COLORS.get(label, NEUTRAL))

    boxes = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.45)
        box.set_edgecolor(color)
    for median in boxes["medians"]:
        median.set_color("black")
    rng = np.random.default_rng(1)
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(
            np.full(values.size, index) + rng.normal(0, 0.06, values.size),
            values, s=16, color=color, edgecolor="white", linewidth=0.4, zorder=3,
        )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{target} expression, log2(normalized counts + 1)")
    top = max((v.max() for v in data), default=1.0)
    _annotate_pairs(ax, expression_pairs, [label.split("\n")[0] for label in labels], top)
    ax.set_title(
        f"A. {target} mRNA levels by histotype\n"
        f"Kruskal-Wallis p = {kruskal.get('p_value', float('nan')):.3f}",
        loc="left",
    )

    ax = axes[1]
    subset = detection[detection["histotype"] != "Tutti"]
    subset = subset.set_index("histotype").reindex([o for o in order if o in set(subset["histotype"])])
    bars = ax.bar(
        range(len(subset)),
        subset["detection_percent"],
        color=[HISTOTYPE_COLORS.get(i, NEUTRAL) for i in subset.index],
        alpha=0.85,
    )
    for rect, (_, row) in zip(bars, subset.iterrows()):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.5,
            f"{int(row['n_positive'])}/{int(row['n'])}",
            ha="center", fontsize=8,
        )
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels(subset.index, fontsize=8)
    ax.set_ylabel(f"{target}-positive samples (%)")
    ax.set_ylim(0, 105)
    _annotate_pairs(ax, detection_pairs, list(subset.index), 100, step=6)
    ax.set_title("B. Detection rate (≥ 5 raw counts)", loc="left")

    fig.tight_layout()
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- #
# Figura 2: stadio, eta', profondita'                                          #
# --------------------------------------------------------------------------- #


def plot_clinical_associations(
    per_sample: pd.DataFrame,
    stage_pairs: pd.DataFrame,
    age_result: Mapping[str, float],
    target: str,
    path: Path,
    dpi: int = 300,
) -> Path | None:
    """Association of the target with FIGO stage and age."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    stages = [s for s in ["I", "II"] if (per_sample["stage"].astype(str) == s).any()]
    data = [
        per_sample.loc[per_sample["stage"].astype(str) == s, "log2_normalized"].to_numpy()
        for s in stages
    ]
    boxes = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.5)
    for box in boxes["boxes"]:
        box.set_facecolor(NEUTRAL)
        box.set_alpha(0.4)
    for median in boxes["medians"]:
        median.set_color("black")
    rng = np.random.default_rng(2)
    for index, values in enumerate(data, start=1):
        ax.scatter(
            np.full(values.size, index) + rng.normal(0, 0.05, values.size),
            values, s=16, color=NEUTRAL, edgecolor="white", linewidth=0.4, zorder=3,
        )
    ax.set_xticks(range(1, len(stages) + 1))
    ax.set_xticklabels([f"FIGO {s}\n(n={d.size})" for s, d in zip(stages, data)], fontsize=8)
    ax.set_ylabel(f"{target}, log2(normalized counts + 1)")
    pvalue = float(stage_pairs["p_value"].iloc[0]) if not stage_pairs.empty else np.nan
    ax.set_title(f"A. {target} by FIGO stage (p = {pvalue:.3f})", loc="left")

    ax = axes[1]
    ax.scatter(
        per_sample["age"], per_sample["log2_normalized"],
        s=22,
        color=[HISTOTYPE_COLORS.get(h, NEUTRAL) for h in per_sample["histotype"]],
        edgecolor="white", linewidth=0.4,
    )
    valid = per_sample.dropna(subset=["age", "log2_normalized"])
    if len(valid) > 3:
        slope, intercept = np.polyfit(valid["age"], valid["log2_normalized"], 1)
        xs = np.linspace(valid["age"].min(), valid["age"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Age at diagnosis (years)")
    ax.set_ylabel(f"{target}, log2(normalized counts + 1)")
    ax.set_title(
        f"B. {target} vs age (Spearman rho = {age_result.get('rho', float('nan')):.2f}, "
        f"p = {age_result.get('p_value', float('nan')):.3f})",
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=6)
        for label, color in HISTOTYPE_COLORS.items()
        if label in set(per_sample["histotype"])
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")

    fig.tight_layout()
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- #
# Figura 3: co-espressione                                                     #
# --------------------------------------------------------------------------- #


def plot_coexpression(
    coexpression: pd.DataFrame,
    panel: pd.DataFrame,
    target: str,
    path: Path,
    dpi: int = 300,
    top_n: int = 20,
) -> Path | None:
    """Panel A: top co-expressed genes. Panel B: validation of the TCGA panel."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))

    ax = axes[0]
    top = coexpression.sort_values("rho", ascending=False).head(top_n).iloc[::-1]
    ax.barh(range(len(top)), top["rho"], color=ACCENT, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.set_xlabel(f"Spearman rho with {target}")
    ax.set_xlim(0, 1)
    ax.set_title(f"A. Top {top_n} genes co-expressed with {target}", loc="left")

    ax = axes[1]
    found = panel[panel["found"]].sort_values("rho")
    if found.empty:
        ax.text(0.5, 0.5, "Nessun gene del pannello disponibile", ha="center")
        ax.axis("off")
    else:
        colors = ["#4C78A8" if s else "#B0B7BF" for s in found["significant"].fillna(False)]
        ax.barh(range(len(found)), found["rho"], color=colors, alpha=0.9)
        ax.errorbar(
            found["rho"], range(len(found)),
            xerr=[found["rho"] - found["ci_low"], found["ci_high"] - found["rho"]],
            fmt="none", ecolor="black", elinewidth=0.7, capsize=2,
        )
        ax.set_yticks(range(len(found)))
        ax.set_yticklabels(found["published_symbol"], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Spearman rho with {target} (95% CI)")
        ax.set_title(
            "B. TCGA/GEPIA2 top-correlated genes,\nvalidation in GSE101108", loc="left"
        )
    fig.tight_layout()
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- #
# Figura 4: espressione differenziale                                          #
# --------------------------------------------------------------------------- #


def plot_volcano(
    table: pd.DataFrame,
    target: str,
    path: Path,
    dpi: int = 300,
    log2fc_threshold: float = 1.0,
    top_labels: int = 12,
) -> Path | None:
    """Volcano plot of the differential genes between positives and negatives."""
    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    logq = -np.log10(table["q_value_BH"].clip(lower=1e-300))
    significant = (table["q_value_BH"] < 0.05) & (table["log2FC"].abs() >= log2fc_threshold)

    ax.scatter(
        table.loc[~significant, "log2FC"], logq[~significant],
        s=8, color="#C8CDD4", alpha=0.6, edgecolor="none",
    )
    up = significant & (table["log2FC"] > 0)
    down = significant & (table["log2FC"] < 0)
    ax.scatter(table.loc[up, "log2FC"], logq[up], s=14, color="#E45756", alpha=0.85,
               edgecolor="none", label=f"up in {target}-positive (n={int(up.sum())})")
    ax.scatter(table.loc[down, "log2FC"], logq[down], s=14, color="#4C78A8", alpha=0.85,
               edgecolor="none", label=f"down in {target}-positive (n={int(down.sum())})")

    ax.axhline(-np.log10(0.05), color="black", linewidth=0.7, linestyle="--")
    ax.axvline(log2fc_threshold, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(-log2fc_threshold, color="black", linewidth=0.5, linestyle=":")

    labelled = table.loc[significant].sort_values("q_value_BH").head(top_labels).copy()
    labelled["logq"] = -np.log10(labelled["q_value_BH"].clip(lower=1e-300))
    _label_volcano(ax, labelled)

    ax.set_xlabel("log2 fold-change (positive vs negative)")
    ax.set_ylabel("-log10 q-value (BH)")
    ax.set_title(f"Differentially expressed genes, {target}-positive vs negative", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    return _save(fig, path, dpi)


def _label_volcano(ax: plt.Axes, labelled: pd.DataFrame) -> None:
    """Label the genes next to their points, with anti-overlap repulsion.

    Uses ``adjustText`` (a ggrepel-like algorithm): the labels stay next to the
    points and are pushed apart just enough not to overlap, with a thin connector
    only where needed.

    """
    texts = [
        ax.text(row["log2FC"], row["logq"], str(gene), fontsize=7,
                ha="center", va="center")
        for gene, row in labelled.iterrows()
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts, ax=ax,
            expand=(1.25, 1.6),
            arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.5),
            force_text=(0.4, 0.7), force_static=(0.2, 0.4),
            only_move={"text": "xy", "static": "xy"},
        )
    except ImportError:  # fallback: piccolo offset fisso
        for t in texts:
            t.set_position((t.get_position()[0], t.get_position()[1] + 0.15))


# --------------------------------------------------------------------------- #
# Figura 5: arricchimento GO                                                   #
# --------------------------------------------------------------------------- #


def plot_go_enrichment(
    table: pd.DataFrame, path: Path, dpi: int = 300, top_n: int = 10
) -> Path | None:
    """Barplot of the most enriched GO terms, per category."""
    if table.empty:
        return None
    categories = [c for c in ["Biological process", "Cellular component", "Molecular function"]
                  if c in set(table["category"])]
    if not categories:
        return None
    fig, axes = plt.subplots(1, len(categories), figsize=(5.2 * len(categories), 4.4))
    if len(categories) == 1:
        axes = [axes]
    letters = "ABC"

    for ax, category, letter in zip(axes, categories, letters):
        subset = (
            table[table["category"] == category]
            .sort_values("p_value")
            .head(top_n)
            .iloc[::-1]
        )
        values = -np.log10(subset["p_value"].clip(lower=1e-300))
        ax.barh(range(len(subset)), values, color="#54A24B", alpha=0.85)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(
            [t if len(t) <= 42 else t[:39] + "..." for t in subset["go_term"]], fontsize=7
        )
        ax.set_xlabel("-log10 p-value")
        ax.set_title(f"{letter}. {category}", loc="left")
    fig.tight_layout()
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- #
# Figura 6: immunologia                                                        #
# --------------------------------------------------------------------------- #


def plot_immune(
    checkpoints: pd.DataFrame,
    signatures: pd.DataFrame,
    target: str,
    path: Path,
    dpi: int = 300,
) -> Path | None:
    """Correlation of the target with immune checkpoints and cell signatures."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    ax = axes[0]
    found = checkpoints[checkpoints["found"]].sort_values("rho")
    colors = ["#4C78A8" if s else "#B0B7BF" for s in found["significant"].fillna(False)]
    ax.barh(range(len(found)), found["rho"], color=colors, alpha=0.9)
    ax.errorbar(
        found["rho"], range(len(found)),
        xerr=[found["rho"] - found["ci_low"], found["ci_high"] - found["rho"]],
        fmt="none", ecolor="black", elinewidth=0.7, capsize=2,
    )
    ax.set_yticks(range(len(found)))
    ax.set_yticklabels(found["published_symbol"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman rho with {target} (95% CI)")
    ax.set_title("A. Immune checkpoint genes", loc="left")

    ax = axes[1]
    ordered = signatures.sort_values("rho")
    colors = ["#F58518" if s else "#B0B7BF" for s in ordered["significant"].fillna(False)]
    ax.barh(range(len(ordered)), ordered["rho"], color=colors, alpha=0.9)
    ax.errorbar(
        ordered["rho"], range(len(ordered)),
        xerr=[ordered["rho"] - ordered["ci_low"], ordered["ci_high"] - ordered["rho"]],
        fmt="none", ecolor="black", elinewidth=0.7, capsize=2,
    )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered["signature"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Spearman rho with {target} (95% CI)")
    ax.set_title("B. Marker-based immune signatures", loc="left")

    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_signature_heatmap(
    scores: pd.DataFrame,
    per_sample: pd.DataFrame,
    order: Sequence[str],
    path: Path,
    dpi: int = 300,
) -> Path | None:
    """Heatmap of the signature scores ordered by target expression."""
    if scores.empty:
        return None
    ordering = per_sample.sort_values(["histotype", "log2_normalized"])
    samples = [s for s in ordering["gsm_id"] if s in scores.index]
    matrix = scores.loc[samples].T

    fig, ax = plt.subplots(figsize=(max(7, len(samples) * 0.11), 4.6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel("Samples (grouped by histotype, sorted by target expression)")

    boundaries, position = [], 0
    for label in order:
        count = int((ordering["histotype"] == label).sum())
        if count:
            boundaries.append((position + count / 2, label))
            position += count
            ax.axvline(position - 0.5, color="black", linewidth=0.6)
    for center, label in boundaries:
        ax.text(center, -0.8, label, ha="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Signature score (z)")
    fig.tight_layout()
    return _save(fig, path, dpi)
