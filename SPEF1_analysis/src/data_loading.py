"""Loading and normalization of the GSE101108 expression data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class DataError(RuntimeError):
    """Dati di ingresso mancanti o incoerenti."""


@dataclass
class Dataset:
    """Aligned counts/metadata dataset.

    Attributes:
        counts: raw counts, genes x samples (columns = GSM IDs).
        normalized: counts normalized by size factor (median-of-ratios).
        log_expression: ``log2(normalized + 1)``.
        metadata: one row per sample, indexed on the GSM ID.
        size_factors: per-sample normalization factors.

    """

    counts: pd.DataFrame
    normalized: pd.DataFrame
    log_expression: pd.DataFrame
    metadata: pd.DataFrame
    size_factors: pd.Series

    @property
    def n_samples(self) -> int:
        return self.counts.shape[1]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[0]

    def subset(self, samples: Sequence[str]) -> "Dataset":
        """Return the dataset restricted to the indicated samples."""
        samples = [s for s in samples if s in self.counts.columns]
        return Dataset(
            counts=self.counts[samples],
            normalized=self.normalized[samples],
            log_expression=self.log_expression[samples],
            metadata=self.metadata.loc[samples],
            size_factors=self.size_factors[samples],
        )


def load_dataset(
    counts_file: Path,
    metadata_file: Path,
    histotype_file: Path,
    histotype_order: Sequence[str],
    include_unspecified: bool = False,
) -> Dataset:
    """Load counts and metadata, align them and normalize them.

    The columns of the matrix produced by the pipeline have the form
    ``GSM2699211|OV106``: they are reduced to the GSM ID alone, while the original
    label stays in the metadata (``sample_label``).

    Raises:
        DataError: if the files do not exist or if no sample can be aligned.

    """
    for path in (counts_file, metadata_file, histotype_file):
        if not Path(path).is_file():
            raise DataError(
                f"File di ingresso mancante: {path}. Eseguire prima "
                f"'python main.py' nella cartella GSE101108_pipeline."
            )

    counts = pd.read_csv(counts_file, index_col=0)
    labels = {c: str(c).split("|")[-1] for c in counts.columns}
    counts.columns = [str(c).split("|")[0] for c in counts.columns]

    metadata = pd.read_excel(metadata_file)
    histotypes = pd.read_excel(histotype_file, sheet_name="classificazione")
    metadata = metadata.merge(
        histotypes[
            [
                "gsm_id",
                "histotype_original",
                "histotype_normalized",
                "histotype_confidence",
                "needs_manual_review",
            ]
        ],
        on="gsm_id",
        how="left",
    )
    metadata["sample_label"] = metadata["gsm_id"].map(
        {gsm: labels.get(gsm, gsm) for gsm in counts.columns}
    )
    metadata = metadata.set_index("gsm_id")

    shared = [c for c in counts.columns if c in metadata.index]
    if not shared:
        raise DataError(
            "Nessun campione in comune fra matrice e metadati: verificare che "
            "la pipeline GSE101108 sia stata eseguita sulla stessa serie."
        )
    counts = counts[shared]
    metadata = metadata.loc[shared]

    if not include_unspecified:
        keep = metadata["histotype_normalized"].isin(histotype_order)
        removed = int((~keep).sum())
        if removed:
            LOGGER.info(
                "Esclusi %d campioni con istotipo non classificabile (%s)",
                removed,
                ", ".join(
                    sorted(metadata.loc[~keep, "histotype_normalized"].astype(str).unique())
                ),
            )
        counts = counts.loc[:, keep.to_numpy()]
        metadata = metadata.loc[keep]

    metadata["histotype"] = pd.Categorical(
        metadata["histotype_normalized"], categories=list(histotype_order), ordered=True
    )
    metadata["age"] = pd.to_numeric(metadata.get("age"), errors="coerce")
    metadata["library_size"] = counts.sum(axis=0)

    size_factors = compute_size_factors(counts)
    normalized = counts.div(size_factors, axis=1)
    log_expression = np.log2(normalized + 1.0)

    LOGGER.info(
        "Dataset caricato: %d geni x %d campioni (%s)",
        counts.shape[0],
        counts.shape[1],
        ", ".join(
            f"{k}={v}" for k, v in metadata["histotype"].value_counts().sort_index().items()
        ),
    )
    return Dataset(counts, normalized, log_expression, metadata, size_factors)


def compute_size_factors(counts: pd.DataFrame) -> pd.Series:
    """Size factors with the median-of-ratios method (Anders & Huber, 2010).

    Uses only the genes with a positive count in every sample, as in DESeq2.

    """
    matrix = counts.to_numpy(dtype=float)
    usable = (matrix > 0).all(axis=1)
    if usable.sum() < 100:
        LOGGER.warning(
            "Solo %d geni utilizzabili per i size factor: si ripiega sulla "
            "normalizzazione per library size totale.",
            int(usable.sum()),
        )
        totals = counts.sum(axis=0)
        return totals / totals.mean()

    log_matrix = np.log(matrix[usable])
    reference = log_matrix.mean(axis=1, keepdims=True)
    factors = np.exp(np.median(log_matrix - reference, axis=0))
    LOGGER.info(
        "Size factor calcolati su %d geni: intervallo %.2f-%.2f",
        int(usable.sum()),
        factors.min(),
        factors.max(),
    )
    return pd.Series(factors, index=counts.columns, name="size_factor")


def filter_expressed_genes(
    dataset: Dataset, min_counts: int, min_fraction: float
) -> pd.Index:
    """Genes with at least ``min_counts`` counts in at least ``min_fraction`` of the samples."""
    detected = (dataset.counts >= min_counts).mean(axis=1)
    keep = detected >= min_fraction
    LOGGER.info(
        "Filtro di espressione (>=%d conteggi nel >=%.0f%% dei campioni): "
        "%d/%d geni mantenuti",
        min_counts,
        min_fraction * 100,
        int(keep.sum()),
        len(keep),
    )
    return dataset.counts.index[keep.to_numpy()]


def resolve_symbols(index: pd.Index, candidates: Sequence[str]) -> str | None:
    """Return the first symbol (or alias) present in the matrix."""
    upper = {str(i).upper(): str(i) for i in index}
    for candidate in candidates:
        key = str(candidate).strip().upper()
        if key in upper:
            return upper[key]
    return None


def target_series(dataset: Dataset, gene: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Raw, normalized and log2 count series of the target gene.

    Raises:
        DataError: if the gene is not present in the matrix.

    """
    symbol = resolve_symbols(dataset.counts.index, [gene])
    if symbol is None:
        raise DataError(
            f"Il gene {gene} non e' presente nella matrice di espressione. "
            f"Verificare l'annotazione genica della pipeline GSE101108."
        )
    return (
        dataset.counts.loc[symbol],
        dataset.normalized.loc[symbol],
        dataset.log_expression.loc[symbol],
    )
