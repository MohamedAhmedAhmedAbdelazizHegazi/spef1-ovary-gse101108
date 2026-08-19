"""Quality controls and diagnostic figures.

Every figure is computed on transformed data (log2) and never on raw untransformed
counts; figures are saved as PNG at 300 dpi.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: required for batch runs on Windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOGGER = logging.getLogger(__name__)

PALETTE = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#EECA3B", "#9D755D", "#BAB0AC",
)


@dataclass
class RunRecorder:
    """Collects errors, warnings and automatic decisions of the pipeline."""

    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    decisions: list[dict[str, str]] = field(default_factory=list)

    def error(self, step: str, message: str) -> None:
        LOGGER.error("[%s] %s", step, message)
        self.errors.append({"step": step, "message": message})

    def warn(self, step: str, message: str) -> None:
        LOGGER.warning("[%s] %s", step, message)
        self.warnings.append({"step": step, "message": message})

    def decide(self, step: str, message: str) -> None:
        LOGGER.info("[%s] %s", step, message)
        self.decisions.append({"step": step, "message": message})

    def as_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "errori": pd.DataFrame(self.errors or [{"step": "", "message": "nessun errore"}]),
            "warning": pd.DataFrame(
                self.warnings or [{"step": "", "message": "nessun warning"}]
            ),
            "decisioni": pd.DataFrame(
                self.decisions or [{"step": "", "message": "nessuna decisione"}]
            ),
        }


# --------------------------------------------------------------------------- #
# Tabelle di QC                                                                #
# --------------------------------------------------------------------------- #


def matrix_overview(matrix: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """General statistics of the expression matrix."""
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    rows = [
        ("tipo_di_dato", data_type),
        ("n_geni", matrix.shape[0]),
        ("n_campioni", matrix.shape[1]),
        ("n_celle", int(values.size)),
        ("n_valori_mancanti", int(np.isnan(values).sum())),
        ("percentuale_mancanti", round(float(np.isnan(values).mean() * 100), 4)),
        ("geni_duplicati", int(matrix.index.duplicated().sum())),
        ("campioni_duplicati", int(matrix.columns.duplicated().sum())),
        ("geni_tutti_zero", int((matrix.fillna(0) == 0).all(axis=1).sum())),
        ("valore_minimo", float(finite.min()) if finite.size else np.nan),
        ("valore_massimo", float(finite.max()) if finite.size else np.nan),
        ("mediana", float(np.median(finite)) if finite.size else np.nan),
    ]
    return pd.DataFrame(rows, columns=["metrica", "valore"])


def library_size_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-sample library size with outlier flagging.

    A sample is flagged if its library size deviates by more than 3 median
    absolute deviations (MAD) from the median.

    """
    totals = matrix.sum(axis=0, skipna=True)
    detected = (matrix.fillna(0) > 0).sum(axis=0)
    median = float(np.median(totals)) if len(totals) else np.nan
    mad = float(np.median(np.abs(totals - median))) if len(totals) else np.nan
    scaled_mad = mad * 1.4826 if mad and not np.isnan(mad) else np.nan
    if not scaled_mad or np.isnan(scaled_mad) or scaled_mad == 0:
        deviation = pd.Series(0.0, index=totals.index)
    else:
        deviation = (totals - median) / scaled_mad

    table = pd.DataFrame(
        {
            "sample_column": totals.index,
            "library_size": totals.to_numpy(),
            "n_features_detected": detected.to_numpy(),
            "robust_z": deviation.round(3).to_numpy(),
        }
    )
    table["outlier"] = table["robust_z"].abs() > 3
    n_outliers = int(table["outlier"].sum())
    if n_outliers:
        LOGGER.warning(
            "%d campioni con library size anomala (|z robusto| > 3): %s",
            n_outliers,
            ", ".join(table.loc[table["outlier"], "sample_column"].astype(str)[:10]),
        )
    return table


def expression_distribution_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-sample expression quantiles (already-transformed data)."""
    described = matrix.describe(percentiles=[0.25, 0.5, 0.75]).transpose()
    described.index.name = "sample_column"
    return described.reset_index()


def histotype_counts_table(metadata: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count of samples per value of the indicated column."""
    counts = (
        metadata[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="n_samples")
    )
    return counts


def duplicated_report(matrix: pd.DataFrame) -> pd.DataFrame:
    """List of duplicate genes and samples in the matrix."""
    dup_genes = matrix.index[matrix.index.duplicated(keep=False)].unique().tolist()
    dup_samples = matrix.columns[matrix.columns.duplicated(keep=False)].unique().tolist()
    rows = [{"tipo": "gene", "valore": g} for g in dup_genes[:1000]]
    rows += [{"tipo": "campione", "valore": s} for s in dup_samples[:1000]]
    return pd.DataFrame(rows or [{"tipo": "", "valore": "nessun duplicato"}])


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path | None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        LOGGER.info("Figura salvata: %s", path.name)
        return path
    except (OSError, PermissionError) as exc:
        LOGGER.error("Impossibile salvare la figura %s: %s", path, exc)
        return None
    finally:
        plt.close(fig)


def plot_library_sizes(
    table: pd.DataFrame, path: Path, dpi: int = 300
) -> Path | None:
    """Histogram and bars of the per-sample library size."""
    if table.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(table["library_size"], bins=min(30, max(5, len(table) // 3)),
                 color=PALETTE[0], edgecolor="white")
    axes[0].set_xlabel("Library size (somma dei conteggi)")
    axes[0].set_ylabel("N campioni")
    axes[0].set_title("Distribuzione della library size")

    ordered = table.sort_values("library_size")
    colors = [PALETTE[3] if o else PALETTE[0] for o in ordered["outlier"]]
    axes[1].bar(range(len(ordered)), ordered["library_size"], color=colors)
    axes[1].set_xlabel("Campioni (ordinati)")
    axes[1].set_ylabel("Library size")
    axes[1].set_title("Library size per campione (rosso = outlier)")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_expression_boxplot(
    log_matrix: pd.DataFrame, path: Path, dpi: int = 300, max_samples: int = 120
) -> Path | None:
    """Boxplot of the per-sample log2 expression distribution."""
    if log_matrix.empty:
        return None
    data = log_matrix.iloc[:, :max_samples]
    fig, ax = plt.subplots(figsize=(max(8, data.shape[1] * 0.16), 5))
    ax.boxplot(
        [data[col].dropna().to_numpy() for col in data.columns],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": PALETTE[0], "alpha": 0.6},
        medianprops={"color": "black"},
    )
    ax.set_xticks(range(1, data.shape[1] + 1))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=6)
    ax.set_ylabel("log2(valore + 1)")
    ax.set_title("Distribuzione dell'espressione per campione")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_selected_genes_heatmap(
    log_matrix: pd.DataFrame,
    sample_labels: Mapping[str, str],
    path: Path,
    dpi: int = 300,
) -> Path | None:
    """Z-score heatmap of the selected genes (rows = genes)."""
    if log_matrix.empty or log_matrix.shape[1] < 2:
        LOGGER.warning("Heatmap non generata: dati insufficienti")
        return None
    values = log_matrix.to_numpy(dtype=float)
    centered = values - np.nanmean(values, axis=1, keepdims=True)
    spread = np.nanstd(values, axis=1, keepdims=True)
    spread[spread == 0] = 1.0
    zscores = centered / spread

    fig, ax = plt.subplots(
        figsize=(max(7, log_matrix.shape[1] * 0.18), max(3, log_matrix.shape[0] * 0.35))
    )
    image = ax.imshow(zscores, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(log_matrix.shape[0]))
    ax.set_yticklabels(log_matrix.index, fontsize=8)
    ax.set_xticks(range(log_matrix.shape[1]))
    ax.set_xticklabels(
        [sample_labels.get(c, c) for c in log_matrix.columns], rotation=90, fontsize=5
    )
    ax.set_title("Geni selezionati - z-score su log2(valore + 1)")
    fig.colorbar(image, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_pca(
    log_matrix: pd.DataFrame,
    groups: Mapping[str, str],
    path: Path,
    dpi: int = 300,
    n_top_genes: int = 2000,
) -> tuple[Path | None, pd.DataFrame | None]:
    """PCA on the samples using the most variable genes (log2 data).

    Returns:
        ``(figure path, coordinates table)``.

    """
    if log_matrix.shape[1] < 3 or log_matrix.shape[0] < 10:
        LOGGER.warning(
            "PCA non calcolata: servono almeno 10 geni e 3 campioni (attuali: %dx%d)",
            *log_matrix.shape,
        )
        return None, None

    data = log_matrix.dropna(how="any")
    if data.shape[0] < 10:
        data = log_matrix.fillna(0.0)
    variances = data.var(axis=1, skipna=True)
    top = variances.sort_values(ascending=False).head(n_top_genes).index
    subset = data.loc[top]

    matrix = subset.to_numpy(dtype=float).T
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        LOGGER.error("PCA fallita: %s", exc)
        return None, None
    scores = u[:, :2] * s[:2]
    explained = (s**2 / np.sum(s**2))[:2] * 100

    coords = pd.DataFrame(
        {
            "sample_column": subset.columns,
            "group": [groups.get(c, "n/d") for c in subset.columns],
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, (label, chunk) in enumerate(coords.groupby("group")):
        ax.scatter(
            chunk["PC1"],
            chunk["PC2"],
            label=f"{label} (n={len(chunk)})",
            color=PALETTE[index % len(PALETTE)],
            s=45,
            edgecolor="white",
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title(f"PCA sui campioni ({len(top)} geni piu' variabili, log2)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path, dpi), coords


def plot_genes_by_histotype(
    log_matrix: pd.DataFrame,
    groups: Mapping[str, str],
    path: Path,
    dpi: int = 300,
    max_genes: int = 12,
) -> Path | None:
    """Boxplot of the selected genes' expression per histotype."""
    if log_matrix.empty:
        return None
    genes = list(log_matrix.index[:max_genes])
    labels = sorted({groups.get(c, "n/d") for c in log_matrix.columns})
    n_cols = min(4, len(genes)) or 1
    n_rows = int(np.ceil(len(genes) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), squeeze=False
    )

    for index, gene in enumerate(genes):
        ax = axes[index // n_cols][index % n_cols]
        series = log_matrix.loc[gene]
        data = [
            series[[c for c in series.index if groups.get(c, "n/d") == label]]
            .dropna()
            .to_numpy()
            for label in labels
        ]
        boxes = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for box_index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(PALETTE[box_index % len(PALETTE)])
            box.set_alpha(0.65)
        for box_index, values in enumerate(data):
            if len(values):
                jitter = np.random.default_rng(0).normal(0, 0.05, len(values))
                ax.scatter(
                    np.full(len(values), box_index + 1) + jitter,
                    values,
                    s=10,
                    color="black",
                    alpha=0.5,
                )
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, fontsize=8, ha="right")
        ax.set_title(str(gene), fontsize=10)
        ax.set_ylabel("log2(valore + 1)", fontsize=8)

    for index in range(len(genes), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")
    fig.suptitle("Espressione dei geni selezionati per istotipo", y=1.01)
    fig.tight_layout()
    return _save(fig, path, dpi)


# --------------------------------------------------------------------------- #
# Assembly of the report
# --------------------------------------------------------------------------- #


def build_summary_table(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Turn the summary dictionary into a two-column table."""
    rows = []
    for key, value in summary.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = "; ".join(f"{k}={v}" for k, v in value.items())
        rows.append({"voce": key, "valore": value})
    return pd.DataFrame(rows)


def summary_text_lines(summary: Mapping[str, Any]) -> list[str]:
    """Text representation of the summary for the .txt file."""
    lines = ["=" * 78, "RIEPILOGO DELLA PIPELINE", "=" * 78]
    for key, value in summary.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = "; ".join(f"{k}={v}" for k, v in value.items())
        lines.append(f"{key:.<45} {value}")
    lines.append("=" * 78)
    return lines


def sample_group_map(
    matching_table: pd.DataFrame, column: str = "histotype"
) -> dict[str, str]:
    """Map ``matrix column -> group label`` for the figures."""
    subset = matching_table.dropna(subset=["matrix_column"])
    return dict(zip(subset["matrix_column"], subset[column].fillna("n/d")))


def sample_label_map(matching_table: pd.DataFrame) -> dict[str, str]:
    """Map ``matrix column -> readable label`` for the figures."""
    subset = matching_table.dropna(subset=["matrix_column"])
    return {
        str(row["matrix_column"]): f"{row['matrix_column']} ({row['gsm_id']})"
        for _, row in subset.iterrows()
    }


def genes_found_summary(lookup: pd.DataFrame) -> dict[str, Iterable[str]]:
    """Lists of found and not-found genes."""
    return {
        "geni_trovati": lookup.loc[lookup["found"], "gene"].tolist(),
        "geni_non_trovati": lookup.loc[~lookup["found"], "gene"].tolist(),
    }


def missing_values_table(matrix: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """Genes with the largest share of missing values."""
    missing = matrix.isna().sum(axis=1)
    frame = (
        missing[missing > 0]
        .sort_values(ascending=False)
        .head(top)
        .rename("n_missing")
        .reset_index()
    )
    if frame.empty:
        return pd.DataFrame([{"gene": "-", "n_missing": 0}])
    frame.columns = ["gene", "n_missing"]
    return frame


def sequence_to_frame(values: Sequence[str], column: str) -> pd.DataFrame:
    """Convert a sequence into a single-column DataFrame (never empty)."""
    return pd.DataFrame({column: list(values) or ["-"]})
