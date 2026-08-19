"""Analyses centered on the target gene (SPEF1).

The statistical design accounts for the fact that SPEF1 is expressed at very
low levels in this cohort: alongside the analysis of expression levels, the
detectability analysis (proportion of positive samples) is always reported,
which is the transcriptomic equivalent of the immunohistochemical positivity
assessed on the tissue microarray.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data_loading import Dataset, resolve_symbols
from .statistics import (
    benjamini_hochberg,
    cliffs_delta,
    describe_by_group,
    effect_size_label,
    kruskal_wallis,
    pairwise_fisher,
    pairwise_mannwhitney,
    spearman_with_ci,
    vectorized_mannwhitney,
    vectorized_spearman,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TargetProfile:
    """Profile of the target gene in the cohort."""

    detection_summary: pd.DataFrame
    detection_sensitivity: pd.DataFrame
    detection_pairwise: pd.DataFrame
    expression_summary: pd.DataFrame
    expression_pairwise: pd.DataFrame
    kruskal: dict[str, float]
    stage: pd.DataFrame
    age: dict[str, float]
    age_groups: pd.DataFrame
    library_size_check: dict[str, float]
    per_sample: pd.DataFrame
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Target profile
# --------------------------------------------------------------------------- #


def detection_table(
    raw_counts: pd.Series, groups: pd.Series, threshold: int, order: Sequence[str]
) -> pd.DataFrame:
    """Proportion of samples with counts >= threshold, per group."""
    positive = raw_counts >= threshold
    rows = []
    for label in order:
        mask = groups == label
        if mask.sum() == 0:
            continue
        subset = positive[mask]
        n = int(mask.sum())
        k = int(subset.sum())
        rows.append(
            {
                "histotype": label,
                "n": n,
                "n_positive": k,
                "detection_rate": k / n,
                "detection_percent": round(100 * k / n, 1),
                "median_counts": float(raw_counts[mask].median()),
                "max_counts": float(raw_counts[mask].max()),
                "n_zero": int((raw_counts[mask] == 0).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    total_n = int(len(raw_counts))
    frame.loc[len(frame)] = {
        "histotype": "Tutti",
        "n": total_n,
        "n_positive": int(positive.sum()),
        "detection_rate": float(positive.mean()),
        "detection_percent": round(100 * float(positive.mean()), 1),
        "median_counts": float(raw_counts.median()),
        "max_counts": float(raw_counts.max()),
        "n_zero": int((raw_counts == 0).sum()),
    }
    return frame


def detection_sensitivity(
    raw_counts: pd.Series,
    groups: pd.Series,
    thresholds: Sequence[int],
    order: Sequence[str],
) -> pd.DataFrame:
    """Tasso di rilevabilita' a diverse soglie (analisi di sensibilita')."""
    rows = []
    for threshold in thresholds:
        positive = raw_counts >= threshold
        row: dict[str, object] = {"threshold_counts": threshold}
        for label in order:
            mask = groups == label
            if mask.sum():
                row[label] = round(100 * float(positive[mask].mean()), 1)
        row["Tutti"] = round(100 * float(positive.mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def profile_target(
    dataset: Dataset,
    raw: pd.Series,
    log_values: pd.Series,
    threshold: int,
    thresholds_sensitivity: Sequence[int],
    order: Sequence[str],
) -> TargetProfile:
    """Full target analysis: detectability, levels, stage, age."""
    groups = dataset.metadata["histotype"].astype(str)
    notes: list[str] = []

    detection = detection_table(raw, groups, threshold, order)
    sensitivity = detection_sensitivity(raw, groups, thresholds_sensitivity, order)
    detection_pairs = pairwise_fisher(raw >= threshold, groups, order)

    expression = describe_by_group(log_values, groups, order)
    overall = describe_by_group(log_values, pd.Series("Tutti", index=log_values.index), ["Tutti"])
    expression = pd.concat([expression, overall], ignore_index=True)
    expression_pairs = pairwise_mannwhitney(log_values, groups, order)
    statistic, pvalue, epsilon = kruskal_wallis(
        [log_values[groups == label].to_numpy() for label in order if (groups == label).any()]
    )
    kruskal = {"H": statistic, "p_value": pvalue, "epsilon_squared": epsilon}
    LOGGER.info(
        "Livelli per istotipo: Kruskal-Wallis H=%.2f p=%.4f (epsilon2=%.3f)",
        statistic,
        pvalue,
        epsilon,
    )

    # stadio FIGO
    stage_groups = dataset.metadata.get("stage")
    if stage_groups is None:
        stage = pd.DataFrame()
        notes.append("Stadio FIGO non disponibile nei metadati.")
    else:
        stage_groups = stage_groups.astype(str).str.strip().str.upper()
        stage_labels = [s for s in ["I", "II"] if (stage_groups == s).any()]
        stage = pairwise_mannwhitney(log_values, stage_groups, stage_labels)
        stage_desc = describe_by_group(log_values, stage_groups, stage_labels)
        stage = stage.merge(
            stage_desc.rename(columns={"group": "group_1"})[["group_1", "n"]],
            on="group_1",
            how="left",
        )

    # eta'
    ages = dataset.metadata["age"]
    if ages.notna().sum() >= 5:
        age = spearman_with_ci(ages, log_values)
        bins = [0, 40, 60, 80, 200]
        labels = ["<=40", "41-60", "61-80", ">80"]
        age_bin = pd.cut(ages, bins=bins, labels=labels, right=True)
        age_groups = pairwise_mannwhitney(log_values, age_bin.astype(str), labels)
    else:
        age = {"rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
        age_groups = pd.DataFrame()
        notes.append("Eta' non disponibile per un numero sufficiente di campioni.")

    # technical control: detectability must not depend on depth
    library_check = spearman_with_ci(dataset.metadata["library_size"], log_values)
    LOGGER.info(
        "Controllo tecnico: correlazione con la library size rho=%.3f (p=%.3f)",
        library_check["rho"],
        library_check["p_value"],
    )
    if library_check["p_value"] < 0.05:
        notes.append(
            "L'espressione del bersaglio correla con la profondita' di "
            "sequenziamento: i confronti sono stati eseguiti su conteggi "
            "normalizzati (median-of-ratios) e il dato va interpretato con cautela."
        )

    per_sample = pd.DataFrame(
        {
            "gsm_id": raw.index,
            "sample_label": dataset.metadata["sample_label"].to_numpy(),
            "histotype": groups.to_numpy(),
            "stage": dataset.metadata.get("stage", pd.Series(index=raw.index)).to_numpy(),
            "age": dataset.metadata["age"].to_numpy(),
            "library_size": dataset.metadata["library_size"].to_numpy(),
            "raw_counts": raw.to_numpy(),
            "normalized_counts": dataset.normalized.loc[raw.name].to_numpy(),
            "log2_normalized": log_values.to_numpy(),
            "positive": (raw >= threshold).to_numpy(),
        }
    )

    return TargetProfile(
        detection_summary=detection,
        detection_sensitivity=sensitivity,
        detection_pairwise=detection_pairs,
        expression_summary=expression,
        expression_pairwise=expression_pairs,
        kruskal=kruskal,
        stage=stage,
        age=age,
        age_groups=age_groups,
        library_size_check=library_check,
        per_sample=per_sample,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Co-espressione                                                               #
# --------------------------------------------------------------------------- #


def coexpression_analysis(
    dataset: Dataset,
    expressed_genes: pd.Index,
    log_values: pd.Series,
    target: str,
) -> pd.DataFrame:
    """Spearman correlation between the target and every expressed gene."""
    matrix = dataset.log_expression.loc[expressed_genes]
    matrix = matrix.drop(index=[target], errors="ignore")
    table = vectorized_spearman(matrix, log_values)
    significant = int((table["q_value_BH"] < 0.05).sum())
    LOGGER.info(
        "Co-espressione: %d geni testati, %d significativi (q<0.05)",
        len(table),
        significant,
    )
    return table


def panel_correlations(
    dataset: Dataset,
    log_values: pd.Series,
    panel: Mapping[str, Sequence[str]] | Sequence[str],
    label: str,
) -> pd.DataFrame:
    """Correlation of the target with a predefined gene panel.

    Args:
        panel: list of symbols or a ``published name -> alias`` map.

    """
    if isinstance(panel, Mapping):
        items = list(panel.items())
    else:
        items = [(gene, [gene]) for gene in panel]

    rows = []
    for published, aliases in items:
        symbol = resolve_symbols(dataset.log_expression.index, aliases)
        if symbol is None:
            rows.append(
                {
                    "panel": label,
                    "published_symbol": published,
                    "symbol_in_matrix": None,
                    "found": False,
                    "median_counts": np.nan,
                    "rho": np.nan,
                    "p_value": np.nan,
                    "n": 0,
                }
            )
            continue
        values = dataset.log_expression.loc[symbol]
        result = spearman_with_ci(values, log_values, n_boot=1000)
        rows.append(
            {
                "panel": label,
                "published_symbol": published,
                "symbol_in_matrix": symbol,
                "found": True,
                "median_counts": float(dataset.counts.loc[symbol].median()),
                "rho": result["rho"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_value": result["p_value"],
                "n": result["n"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_value_BH"] = benjamini_hochberg(frame["p_value"])
        frame["significant"] = frame["q_value_BH"] < 0.05
        found = int(frame["found"].sum())
        LOGGER.info(
            "Pannello '%s': %d/%d geni trovati, %d correlazioni significative",
            label,
            found,
            len(frame),
            int(frame["significant"].fillna(False).sum()),
        )
    return frame


# --------------------------------------------------------------------------- #
# Espressione differenziale                                                    #
# --------------------------------------------------------------------------- #


def stratified_correlations(
    dataset: Dataset,
    log_values: pd.Series,
    features: Mapping[str, pd.Series],
    order: Sequence[str],
    min_samples: int = 8,
) -> pd.DataFrame:
    """Correlations repeated within each histotype (confounding control).

    If an association holds within the individual histotypes it cannot be
    explained by the histological composition of the cohort alone.

    Args:
        features: ``name -> value series`` map (genes or signature scores).
        min_samples: minimum size to compute the correlation.

    """
    groups = dataset.metadata["histotype"].astype(str)
    rows = []
    for name, values in features.items():
        for label in list(order) + ["All histotypes"]:
            if label == "All histotypes":
                mask = pd.Series(True, index=groups.index)
            else:
                mask = groups == label
            if int(mask.sum()) < min_samples:
                continue
            result = spearman_with_ci(values[mask], log_values[mask], n_boot=1000)
            rows.append(
                {
                    "feature": name,
                    "histotype": label,
                    "n": int(mask.sum()),
                    "n_used": result["n"],
                    "estimable": not np.isnan(result["rho"]),
                    "rho": result["rho"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "p_value": result["p_value"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["q_value_BH"] = benjamini_hochberg(frame["p_value"])
        frame["significant"] = frame["q_value_BH"] < 0.05
        LOGGER.info(
            "Analisi stratificata per istotipo: %d combinazioni, %d significative",
            len(frame),
            int(frame["significant"].fillna(False).sum()),
        )
    return frame


def differential_expression(
    dataset: Dataset,
    expressed_genes: pd.Index,
    grouping: pd.Series,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Differential expression between two sample groups.

    ``grouping`` is a label series indexed on the GSM IDs.

    """
    samples_a = grouping.index[grouping == label_a].tolist()
    samples_b = grouping.index[grouping == label_b].tolist()
    LOGGER.info(
        "Espressione differenziale %s (n=%d) vs %s (n=%d) su %d geni",
        label_a,
        len(samples_a),
        label_b,
        len(samples_b),
        len(expressed_genes),
    )
    if min(len(samples_a), len(samples_b)) < 3:
        LOGGER.warning(
            "Gruppi troppo piccoli per l'analisi differenziale: risultato omesso"
        )
        return pd.DataFrame()

    matrix = dataset.log_expression.loc[expressed_genes]
    table = vectorized_mannwhitney(matrix, samples_a, samples_b)
    table = table.rename(
        columns={
            "median_group_a": f"median_{label_a}",
            "median_group_b": f"median_{label_b}",
            "mean_group_a": f"mean_{label_a}",
            "mean_group_b": f"mean_{label_b}",
        }
    )
    table["direction"] = np.where(table["log2FC"] > 0, "up", "down")
    significant = int((table["q_value_BH"] < 0.05).sum())
    LOGGER.info("Geni differenziali (q<0.05): %d", significant)
    return table


def split_by_target(
    raw: pd.Series, log_values: pd.Series, threshold: int
) -> tuple[pd.Series, pd.Series]:
    """Two stratification criteria: detectability and median.

    Returns:
        ``(group by detectability, group by median)``.

    """
    detection = pd.Series(
        np.where(raw >= threshold, "positive", "negative"), index=raw.index
    )
    median = float(log_values.median())
    median_split = pd.Series(
        np.where(log_values > median, "high", "low"), index=log_values.index
    )
    LOGGER.info(
        "Stratificazione: positivi=%d negativi=%d (soglia %d conteggi); "
        "high=%d low=%d (mediana log2=%.2f)",
        int((detection == "positive").sum()),
        int((detection == "negative").sum()),
        threshold,
        int((median_split == "high").sum()),
        int((median_split == "low").sum()),
        median,
    )
    return detection, median_split


# --------------------------------------------------------------------------- #
# Immunologia                                                                  #
# --------------------------------------------------------------------------- #


def signature_scores(
    dataset: Dataset, signatures: Mapping[str, Sequence[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marker-based signature scores (mean of the marker z-scores).

    This is not a deconvolution of the infiltrate: it is a relative estimate based
    on canonical markers, to be declared as such.

    Returns:
        ``(sample x signature scores, signature composition)``.

    """
    scores = {}
    composition = []
    for name, genes in signatures.items():
        found = [g for g in genes if resolve_symbols(dataset.log_expression.index, [g])]
        symbols = [resolve_symbols(dataset.log_expression.index, [g]) for g in found]
        composition.append(
            {
                "signature": name,
                "n_markers_requested": len(genes),
                "n_markers_found": len(symbols),
                "markers_found": ", ".join(symbols),
                "markers_missing": ", ".join(sorted(set(genes) - set(found))),
            }
        )
        if not symbols:
            LOGGER.warning("Firma '%s': nessun marcatore trovato, saltata", name)
            continue
        block = dataset.log_expression.loc[symbols]
        zscores = block.sub(block.mean(axis=1), axis=0).div(
            block.std(axis=1).replace(0, np.nan), axis=0
        )
        scores[name] = zscores.mean(axis=0)
    return pd.DataFrame(scores), pd.DataFrame(composition)


def immune_associations(
    dataset: Dataset,
    log_values: pd.Series,
    checkpoints: Sequence[str],
    scores: pd.DataFrame,
    grouping: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Association of the target with immune checkpoints and signatures.

    Returns:
        ``(checkpoint table, signature table)``.

    """
    checkpoint_table = panel_correlations(
        dataset, log_values, list(checkpoints), "immune checkpoints"
    )

    rows = []
    for name in scores.columns:
        values = scores[name]
        result = spearman_with_ci(values, log_values, n_boot=1000)
        positive = values[grouping == "positive"].dropna()
        negative = values[grouping == "negative"].dropna()
        delta = cliffs_delta(positive, negative)
        rows.append(
            {
                "signature": name,
                "rho": result["rho"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "p_value": result["p_value"],
                "median_positive": float(positive.median()) if positive.size else np.nan,
                "median_negative": float(negative.median()) if negative.size else np.nan,
                "cliffs_delta": delta,
                "effect_size": effect_size_label(delta),
            }
        )
    signature_table = pd.DataFrame(rows)
    if not signature_table.empty:
        signature_table["q_value_BH"] = benjamini_hochberg(signature_table["p_value"])
        signature_table["significant"] = signature_table["q_value_BH"] < 0.05
    return checkpoint_table, signature_table
