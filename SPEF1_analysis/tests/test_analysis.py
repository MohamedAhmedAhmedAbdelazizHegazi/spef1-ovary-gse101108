"""Unit tests of the SPEF1 analysis.

Run::

    python -m pytest tests -v

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analyses, data_loading, enrichment, statistics  # noqa: E402


# --------------------------------------------------------------------------- #
# Statistica                                                                   #
# --------------------------------------------------------------------------- #


def test_benjamini_hochberg_monotone_and_bounded():
    pvalues = [0.001, 0.008, 0.039, 0.041, 0.9]
    q = statistics.benjamini_hochberg(pvalues)
    assert np.all(np.diff(q) >= -1e-12)  # non-decreasing monotonicity
    assert np.all(q <= 1.0) and np.all(q >= 0.0)
    assert q[0] == pytest.approx(0.005, abs=1e-9)


def test_benjamini_hochberg_propagates_nan():
    q = statistics.benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(q[1])
    assert not np.isnan(q[0])


def test_cliffs_delta_extremes():
    assert statistics.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert statistics.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert statistics.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_effect_size_labels():
    assert statistics.effect_size_label(0.05) == "trascurabile"
    assert statistics.effect_size_label(0.6) == "grande"
    assert statistics.effect_size_label(np.nan) == "n/d"


def test_spearman_guard_on_low_variability():
    """A gene at zero in almost every sample must not produce rho=1."""
    almost_constant = [0, 0, 0, 0, 0, 0, 0, 3.0]
    other = [1.0, 2, 3, 4, 5, 6, 7, 8]
    result = statistics.spearman_with_ci(almost_constant, other, n_boot=50)
    assert np.isnan(result["rho"])

    variable = [1.0, 2, 3, 4, 5, 6, 7, 9]
    ok = statistics.spearman_with_ci(variable, other, n_boot=50)
    assert ok["rho"] == pytest.approx(1.0)


def test_vectorized_spearman_matches_scipy():
    from scipy import stats as sp

    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(rng.normal(size=(5, 30)), index=[f"G{i}" for i in range(5)])
    reference = pd.Series(rng.normal(size=30), index=matrix.columns)
    table = statistics.vectorized_spearman(matrix, reference)
    expected = sp.spearmanr(matrix.loc["G2"], reference)
    assert table.loc["G2", "rho"] == pytest.approx(expected.statistic, abs=1e-10)
    assert table.loc["G2", "p_value"] == pytest.approx(expected.pvalue, abs=1e-8)


def test_vectorized_mannwhitney_log2fc_sign():
    matrix = pd.DataFrame(
        {"a1": [3.0, 1.0], "a2": [3.2, 1.1], "b1": [1.0, 3.0], "b2": [1.1, 3.1]},
        index=["UP", "DOWN"],
    )
    table = statistics.vectorized_mannwhitney(matrix, ["a1", "a2"], ["b1", "b2"])
    assert table.loc["UP", "log2FC"] > 0
    assert table.loc["DOWN", "log2FC"] < 0
    assert table.loc["UP", "cliffs_delta"] == pytest.approx(1.0)


def test_pairwise_tests_apply_fdr():
    values = pd.Series([1, 2, 3, 8, 9, 10, 1, 1, 2], index=[f"S{i}" for i in range(9)])
    groups = pd.Series(["A"] * 3 + ["B"] * 3 + ["C"] * 3, index=values.index)
    table = statistics.pairwise_mannwhitney(values, groups, ["A", "B", "C"])
    assert len(table) == 3
    assert "q_value_BH" in table.columns
    assert (table["q_value_BH"] >= table["p_value"] - 1e-12).all()


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def test_size_factors_track_sequencing_depth():
    """A sample sequenced at twice the depth has a size factor of ~2."""
    base = np.array([100.0, 200, 50, 400, 25, 80, 150, 300])
    counts = pd.DataFrame(
        {"A": base, "B": base * 2, "C": base * 0.5},
        index=[f"G{i}" for i in range(len(base))],
    )
    factors = data_loading.compute_size_factors(counts)
    assert factors["B"] / factors["A"] == pytest.approx(2.0, rel=1e-6)
    assert factors["C"] / factors["A"] == pytest.approx(0.5, rel=1e-6)


def test_filter_expressed_genes():
    counts = pd.DataFrame(
        {"S1": [100, 0, 5], "S2": [120, 0, 6], "S3": [90, 1, 4], "S4": [110, 0, 20]},
        index=["HIGH", "OFF", "LOW"],
    )
    dataset = data_loading.Dataset(
        counts=counts,
        normalized=counts,
        log_expression=np.log2(counts + 1),
        metadata=pd.DataFrame(index=counts.columns),
        size_factors=pd.Series(1.0, index=counts.columns),
    )
    kept = data_loading.filter_expressed_genes(dataset, min_counts=10, min_fraction=0.5)
    assert list(kept) == ["HIGH"]


def test_resolve_symbols_uses_aliases():
    index = pd.Index(["TP53", "CFAP77", "FOXJ1"])
    assert data_loading.resolve_symbols(index, ["C9ORF171", "CFAP77"]) == "CFAP77"
    assert data_loading.resolve_symbols(index, ["NOT_A_GENE"]) is None


# --------------------------------------------------------------------------- #
# Target analysis
# --------------------------------------------------------------------------- #


def test_detection_table_counts_positives():
    raw = pd.Series([0, 2, 5, 20, 0, 7], index=[f"S{i}" for i in range(6)])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"], index=raw.index)
    table = analyses.detection_table(raw, groups, threshold=5, order=["A", "B"])
    row_a = table[table["histotype"] == "A"].iloc[0]
    assert row_a["n"] == 3 and row_a["n_positive"] == 1
    assert table[table["histotype"] == "Tutti"].iloc[0]["n_positive"] == 3


def test_detection_sensitivity_is_monotone():
    raw = pd.Series([0, 1, 3, 5, 10, 30], index=[f"S{i}" for i in range(6)])
    groups = pd.Series(["A"] * 6, index=raw.index)
    table = analyses.detection_sensitivity(raw, groups, [1, 3, 5, 10], ["A"])
    assert list(table["Tutti"]) == sorted(table["Tutti"], reverse=True)


def test_split_by_target_labels():
    raw = pd.Series([0, 10, 2, 8], index=list("abcd"))
    log_values = np.log2(raw + 1)
    detection, median_split = analyses.split_by_target(raw, log_values, threshold=5)
    assert detection.to_dict() == {
        "a": "negative", "b": "positive", "c": "negative", "d": "positive"
    }
    assert set(median_split.unique()) <= {"high", "low"}


# --------------------------------------------------------------------------- #
# Arricchimento                                                                #
# --------------------------------------------------------------------------- #


def _toy_ontology() -> enrichment.GeneOntology:
    background = {f"G{i}" for i in range(100)} | {"CIL1", "CIL2", "CIL3", "CIL4", "CIL5", "CIL6"}
    rows = [
        {"symbol": g, "go_id": "GO:0005930", "go_term": "axoneme", "category": "Component"}
        for g in ["CIL1", "CIL2", "CIL3", "CIL4", "CIL5", "CIL6"]
    ]
    rows += [
        {"symbol": f"G{i}", "go_id": "GO:0000001", "go_term": "other", "category": "Component"}
        for i in range(40)
    ]
    return enrichment.GeneOntology(pd.DataFrame(rows), background)


def test_overrepresentation_detects_enriched_term():
    ontology = _toy_ontology()
    table = enrichment.overrepresentation(
        ["CIL1", "CIL2", "CIL3", "CIL4", "G1", "G2"], ontology, min_genes=3, max_genes=50
    )
    top = table.sort_values("p_value").iloc[0]
    assert top["go_term"] == "axoneme"
    assert top["n_overlap"] == 4
    assert top["fold_enrichment"] > 5
    assert top["q_value_BH"] < 0.05


def test_overrepresentation_ignores_genes_outside_background():
    ontology = _toy_ontology()
    table = enrichment.overrepresentation(["NOT_IN_UNIVERSE"], ontology, min_genes=3)
    assert table.empty


def test_signature_scores_are_centred():
    genes = ["CD68", "CD163", "OTHER"]
    values = pd.DataFrame(
        np.array([[1.0, 5.0, 3.0], [2.0, 6.0, 4.0], [9.0, 1.0, 5.0]]),
        index=genes,
        columns=["S1", "S2", "S3"],
    )
    dataset = data_loading.Dataset(
        counts=values, normalized=values, log_expression=values,
        metadata=pd.DataFrame(index=values.columns),
        size_factors=pd.Series(1.0, index=values.columns),
    )
    scores, composition = analyses.signature_scores(dataset, {"Macrophages": ["CD68", "CD163"]})
    assert scores.shape == (3, 1)
    assert scores["Macrophages"].mean() == pytest.approx(0.0, abs=1e-9)
    assert composition.iloc[0]["n_markers_found"] == 2
