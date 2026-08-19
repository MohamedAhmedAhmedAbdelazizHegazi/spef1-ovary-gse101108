"""Essential unit tests of the GSE101108 pipeline.

Run::

    python -m pytest tests -v

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import expression_loader, filtering, gene_annotation, geo_metadata  # noqa: E402
from src import sample_matching  # noqa: E402


# --------------------------------------------------------------------------- #
# Histotype normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("clear cell", filtering.CLEAR_CELL),
        ("clear-cell", filtering.CLEAR_CELL),
        ("clear_cell", filtering.CLEAR_CELL),
        ("Ovarian clear cell carcinoma", filtering.CLEAR_CELL),
        ("OCCC", filtering.CLEAR_CELL),
        ("CCC", filtering.CLEAR_CELL),
        ("endometrioid", filtering.ENDOMETRIOID),
        ("Endometrioid carcinoma", filtering.ENDOMETRIOID),
        ("ovarian endometrioid carcinoma", filtering.ENDOMETRIOID),
        ("EC", filtering.ENDOMETRIOID),
        ("mucinous", filtering.MUCINOUS),
        ("Mucinous carcinoma", filtering.MUCINOUS),
        ("MC", filtering.MUCINOUS),
        ("serous", filtering.SEROUS),
        ("High-grade serous carcinoma", filtering.SEROUS),
        ("HGSC", filtering.SEROUS),
        ("LGSC", filtering.SEROUS),
        ("", filtering.UNSPECIFIED),
        ("NA", filtering.UNSPECIFIED),
        (None, filtering.UNSPECIFIED),
        ("Metastasis", filtering.UNSPECIFIED),
        ("qualcosa di ignoto", filtering.UNSPECIFIED),
    ],
)
def test_normalize_histotype(value, expected):
    assert filtering.normalize_histotype(value)[0] == expected


def test_serous_wins_over_other_terms():
    """A value containing 'serous' is always classified as Serous."""
    label, rule, _, _ = filtering.normalize_histotype("mixed serous and endometrioid")
    assert label == filtering.SEROUS
    assert rule == "serous_text"


def test_unrecognized_values_are_flagged_for_review():
    _, _, confidence, review = filtering.normalize_histotype("adenocarcinoma NOS")
    assert review is True
    assert confidence in {"low", "medium"}


def test_build_sample_sets_keeps_unspecified_out_of_final_dataset():
    frame = pd.DataFrame(
        {
            "gsm_id": ["G1", "G2", "G3", "G4"],
            "histotype_normalized": [
                filtering.CLEAR_CELL,
                filtering.SEROUS,
                filtering.UNSPECIFIED,
                filtering.MUCINOUS,
            ],
            "needs_manual_review": [False, False, True, False],
        }
    )
    sets = filtering.build_sample_sets(
        frame, ["Clear cell", "Endometrioid", "Mucinous"], ["Serous"]
    )
    assert len(sets.all_samples) == 4
    assert len(sets.non_serous) == 3  # 'Other or unspecified' stays non-serous
    assert sorted(sets.allowed["gsm_id"]) == ["G1", "G4"]
    assert sets.to_review["gsm_id"].tolist() == ["G3"]


# --------------------------------------------------------------------------- #
# Identificativi genici                                                        #
# --------------------------------------------------------------------------- #


def test_non_serous_subset_exists_even_when_serous_is_allowed():
    """Including the serous carcinomas must not make the non-serous subset disappear."""
    frame = pd.DataFrame(
        {
            "gsm_id": ["G1", "G2", "G3"],
            "histotype_normalized": [
                filtering.CLEAR_CELL,
                filtering.SEROUS,
                filtering.UNSPECIFIED,
            ],
            "needs_manual_review": [False, False, True],
        }
    )
    sets = filtering.build_sample_sets(
        frame, ["Clear cell", "Endometrioid", "Mucinous", "Serous"], []
    )
    assert sorted(sets.allowed["gsm_id"]) == ["G1", "G2"]  # sierose incluse
    assert sorted(sets.non_serous["gsm_id"]) == ["G1", "G3"]  # sottoinsieme intatto


def test_excluded_histotypes_win_over_allowed():
    frame = pd.DataFrame(
        {
            "gsm_id": ["G1", "G2"],
            "histotype_normalized": [filtering.CLEAR_CELL, filtering.SEROUS],
            "needs_manual_review": [False, False],
        }
    )
    sets = filtering.build_sample_sets(frame, ["Clear cell", "Serous"], ["Serous"])
    assert sets.allowed["gsm_id"].tolist() == ["G1"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ENSG00000141510.17", "ENSG00000141510"),
        ("ENSG00000141510", "ENSG00000141510"),
        ("ensg00000141510.2", "ENSG00000141510"),
        ("ENSMUSG00000059552.9", "ENSMUSG00000059552"),
        ("TP53", "TP53"),
        ("7157", "7157"),
    ],
)
def test_clean_ensembl_id(raw, expected):
    assert gene_annotation.clean_ensembl_id(raw) == expected


def test_detect_id_type():
    assert (
        gene_annotation.detect_id_type(["ENSG00000141510", "ENSG00000012048"])
        == gene_annotation.ID_TYPE_ENSEMBL
    )
    assert (
        gene_annotation.detect_id_type(["TP53", "BRCA1", "KRAS"])
        == gene_annotation.ID_TYPE_SYMBOL
    )
    assert (
        gene_annotation.detect_id_type(["7157", "672", "3845"])
        == gene_annotation.ID_TYPE_ENTREZ
    )


# --------------------------------------------------------------------------- #
# Gene column
# --------------------------------------------------------------------------- #


def test_detect_gene_column_by_name():
    frame = pd.DataFrame({"S1": [1, 2], "gene_symbol": ["TP53", "KRAS"], "S2": [3, 4]})
    candidates = ["gene_symbol", "symbol", "gene", "gene_id"]
    assert expression_loader.detect_gene_column(frame.columns, candidates, frame) == (
        "gene_symbol"
    )


def test_detect_gene_column_by_content():
    """The gene column is not always the first and may have an atypical name."""
    frame = pd.DataFrame(
        {
            "S1": [10, 20, 30],
            "feature_identifier": ["ENSG00000141510", "ENSG00000012048", "ENSG00000133703"],
            "S2": [1, 2, 3],
        }
    )
    detected = expression_loader.detect_gene_column(
        frame.columns, ["gene_symbol", "symbol"], frame
    )
    assert detected == "feature_identifier"


def test_detect_separator():
    assert expression_loader.detect_separator(["a\tb\tc", "1\t2\t3"]) == "\t"
    assert expression_loader.detect_separator(["a,b,c", "1,2,3"]) == ","


# --------------------------------------------------------------------------- #
# Sample matching
# --------------------------------------------------------------------------- #


def test_match_samples_exact_and_token():
    metadata = pd.DataFrame(
        {
            "gsm_id": ["GSM1", "GSM2", "GSM3"],
            "title": ["Tumor_OV106", "Tumor_OV131", "Tumor_OV135"],
            "source_name": ["ovarian carcinoma"] * 3,
            "histotype_normalized": ["Clear cell", "Mucinous", "Serous"],
        }
    )
    result = sample_matching.match_samples_to_columns(
        metadata, ["OV106", "GSM2", "OV135"]
    )
    assert result.matched == {"GSM1": "OV106", "GSM2": "GSM2", "GSM3": "OV135"}
    assert result.unmatched_samples == []
    assert result.unmatched_columns == []
    assert result.match_fraction == 1.0


def test_match_samples_reports_unmatched():
    metadata = pd.DataFrame(
        {
            "gsm_id": ["GSM1", "GSM2"],
            "title": ["Tumor_OV106", "Tumor_OV131"],
            "histotype_normalized": ["Clear cell", "Mucinous"],
        }
    )
    result = sample_matching.match_samples_to_columns(metadata, ["OV106", "OV999"])
    assert result.matched == {"GSM1": "OV106"}
    assert result.unmatched_samples == ["GSM2"]
    assert result.unmatched_columns == ["OV999"]


def test_ambiguous_matches_are_not_assigned():
    """Two samples claiming the same column stay unmatched."""
    metadata = pd.DataFrame(
        {
            "gsm_id": ["GSM1", "GSM2"],
            "title": ["OV10", "OV10"],
            "histotype_normalized": ["Clear cell", "Clear cell"],
        }
    )
    result = sample_matching.match_samples_to_columns(metadata, ["OV10"])
    assert result.matched == {}
    assert result.conflicts.shape[0] == 2


def test_check_match_coverage_threshold():
    metadata = pd.DataFrame(
        {
            "gsm_id": ["GSM1", "GSM2"],
            "title": ["OV1", "OV2"],
            "histotype_normalized": ["Clear cell", "Mucinous"],
        }
    )
    result = sample_matching.match_samples_to_columns(metadata, ["OV1"])
    ok, fraction, _ = sample_matching.check_match_coverage(
        result, ["GSM1", "GSM2"], 0.8, force=False
    )
    assert fraction == 0.5 and ok is False
    forced, _, _ = sample_matching.check_match_coverage(
        result, ["GSM1", "GSM2"], 0.8, force=True
    )
    assert forced is True


# --------------------------------------------------------------------------- #
# Aggregation of duplicates
# --------------------------------------------------------------------------- #


def test_aggregate_duplicates_sum_for_counts():
    matrix = pd.DataFrame(
        {"S1": [10, 5, 3], "S2": [20, 1, 7]},
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    result = gene_annotation.aggregate_duplicate_genes(
        matrix, ["TP53", "TP53", "KRAS"], is_count_like=True
    )
    assert result.method == "sum"
    assert result.matrix.loc["TP53", "S1"] == 15
    assert result.matrix.loc["TP53", "S2"] == 21
    assert result.n_rows_after == 2


def test_aggregate_duplicates_mean_for_tpm():
    matrix = pd.DataFrame({"S1": [10.0, 20.0]}, index=["E1", "E2"])
    result = gene_annotation.aggregate_duplicate_genes(
        matrix, ["TP53", "TP53"], is_count_like=False
    )
    assert result.method == "mean"
    assert result.matrix.loc["TP53", "S1"] == 15.0


def test_extract_genes_of_interest_reports_missing():
    matrix = pd.DataFrame({"S1": [1.0, 2.0]}, index=["TP53", "KRAS"])
    annotation = pd.DataFrame(
        {
            "original_id": ["ENSG1", "ENSG2"],
            "gene_symbol": ["TP53", "KRAS"],
        }
    )
    selected, lookup = gene_annotation.extract_genes_of_interest(
        matrix, ["tp53", "SPEF1"], annotation
    )
    assert list(selected.index) == ["TP53"]
    assert lookup.loc[lookup["gene"] == "SPEF1", "found"].item() is False


# --------------------------------------------------------------------------- #
# Metadati                                                                     #
# --------------------------------------------------------------------------- #


def test_parse_characteristics_handles_variants():
    parsed = geo_metadata.parse_characteristics(
        [
            "histotype: clear cell",
            "FIGO stage: I",
            "Grade : 3",
            "age: 61",
            "treatment: none",
            "senza separatore",
        ]
    )
    assert parsed["histotype"] == "clear cell"
    assert parsed["stage"] == "I"
    assert parsed["grade"] == "3"
    assert parsed["age"] == "61"
    assert "senza separatore" in parsed.values()


def test_parse_characteristics_merges_duplicate_keys():
    parsed = geo_metadata.parse_characteristics(
        ["treatment: chemo", "treatment: surgery"]
    )
    assert parsed["treatment"] == "chemo | surgery"


def test_find_histotype_column():
    frame = pd.DataFrame(columns=["gsm_id", "Histological Type", "age"])
    assert (
        geo_metadata.find_histotype_column(frame, ["histotype", "histological type"])
        == "Histological Type"
    )
    empty = pd.DataFrame(columns=["gsm_id", "age"])
    assert geo_metadata.find_histotype_column(empty, ["histotype"]) is None
