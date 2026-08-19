"""Histotype normalization and sample selection.

The classification is based on explicit rules: each sample keeps its original
value, the normalized value, the rule applied, a confidence level and a manual
review flag. No sample is dropped without being recorded.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

CLEAR_CELL = "Clear cell"
ENDOMETRIOID = "Endometrioid"
MUCINOUS = "Mucinous"
SEROUS = "Serous"
UNSPECIFIED = "Other or unspecified"

#: Valori considerati mancanti.
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not available", "-"}


@dataclass(frozen=True)
class HistotypeRule:
    """Histotype normalization rule.

    Attributes:
        name: human-readable identifier of the rule (reported in the reports).
        pattern: regular expression applied to the normalized value.
        label: canonical histotype assigned.
        confidence: ``high`` | ``medium`` | ``low``.
        needs_review: if ``True`` the sample ends up among those to review.

    """

    name: str
    pattern: str
    label: str
    confidence: str = "high"
    needs_review: bool = False


#: Order matters: the first matching rule wins.
#: Serous carcinomas are evaluated first because 'serous' always excludes.
HISTOTYPE_RULES: tuple[HistotypeRule, ...] = (
    HistotypeRule("serous_text", r"serous", SEROUS),
    HistotypeRule("serous_abbrev", r"^(hgsc|lgsc|hgs|lgs|sc|soc)$", SEROUS),
    HistotypeRule(
        "clear_cell_text", r"clear[\s\-_]*cell", CLEAR_CELL
    ),
    HistotypeRule("clear_cell_abbrev", r"^(occc|ccc|ccoc|cc)$", CLEAR_CELL),
    HistotypeRule("endometrioid_text", r"endometri?oid", ENDOMETRIOID),
    HistotypeRule("endometrioid_abbrev", r"^(ec|eoc|oec|emc)$", ENDOMETRIOID),
    HistotypeRule("mucinous_text", r"mucinous|mucinos[ao]", MUCINOUS),
    HistotypeRule("mucinous_abbrev", r"^(mc|moc|omc)$", MUCINOUS),
    HistotypeRule(
        "metastasis",
        r"metasta",
        UNSPECIFIED,
        confidence="high",
        needs_review=True,
    ),
    HistotypeRule(
        "mixed_or_undifferentiated",
        r"mixed|undifferentiated|carcinosarcoma|not otherwise specified|\bnos\b|"
        r"adenocarcinoma$",
        UNSPECIFIED,
        confidence="medium",
        needs_review=True,
    ),
)


def _normalize_text(value: object) -> str:
    """Minuscole, separatori uniformati, spazi compressi."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[\-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_histotype(value: object) -> tuple[str, str, str, bool]:
    """Normalize a single histotype value.

    Args:
        value: original value (string, ``None`` or NaN).

    Returns:
        ``(label, rule_name, confidence, needs_review)``.

    Examples:
        >>> normalize_histotype("ovarian clear cell carcinoma")[0]
        'Clear cell'
        >>> normalize_histotype("CCC")[0]
        'Clear cell'
        >>> normalize_histotype("HGSC")[0]
        'Serous'
        >>> normalize_histotype("something unknown")[0]
        'Other or unspecified'

    """
    text = _normalize_text(value)
    if text in MISSING_TOKENS:
        return UNSPECIFIED, "missing_value", "high", True

    # valori multipli separati da '|' o ','
    parts = [p.strip() for p in re.split(r"\||,|;", text) if p.strip()]
    labels: list[tuple[str, HistotypeRule]] = []
    for part in parts or [text]:
        compact = re.sub(r"\s+", "", part)
        for rule in HISTOTYPE_RULES:
            target = compact if rule.name.endswith("abbrev") else part
            if re.search(rule.pattern, target):
                labels.append((rule.label, rule))
                break

    if not labels:
        return UNSPECIFIED, "no_rule_matched", "low", True

    distinct = {label for label, _ in labels}
    if len(distinct) > 1:
        return (
            UNSPECIFIED,
            "ambiguous_multiple_labels:" + "+".join(sorted(distinct)),
            "low",
            True,
        )
    label, rule = labels[0]
    return label, rule.name, rule.confidence, rule.needs_review


def classify_histotypes(
    metadata: pd.DataFrame, histotype_column: str | None
) -> pd.DataFrame:
    """Add the histotype classification columns to the metadata.

    Args:
        metadata: table of normalized metadata (one row per GSM).
        histotype_column: name of the column holding the raw histotype;
            if ``None`` every sample becomes ``Other or unspecified``.

    Returns:
        Copy of ``metadata`` with the ``histotype_original``,
        ``histotype_normalized``, ``histotype_rule``, ``histotype_confidence``,
        ``needs_manual_review`` columns.

    """
    frame = metadata.copy()
    if histotype_column is None or histotype_column not in frame.columns:
        LOGGER.error(
            "Istotipo non identificabile: tutti i campioni saranno classificati "
            "come '%s' e nessun dataset per istotipo potra' essere prodotto.",
            UNSPECIFIED,
        )
        frame["histotype_original"] = pd.NA
        frame["histotype_normalized"] = UNSPECIFIED
        frame["histotype_rule"] = "column_not_found"
        frame["histotype_confidence"] = "low"
        frame["needs_manual_review"] = True
        return frame

    originals = frame[histotype_column]
    results = [normalize_histotype(value) for value in originals]
    frame["histotype_original"] = originals
    frame["histotype_normalized"] = [r[0] for r in results]
    frame["histotype_rule"] = [r[1] for r in results]
    frame["histotype_confidence"] = [r[2] for r in results]
    frame["needs_manual_review"] = [r[3] for r in results]

    counts = frame["histotype_normalized"].value_counts()
    LOGGER.info("Distribuzione degli istotipi normalizzati:")
    for label, count in counts.items():
        LOGGER.info("  %-22s %3d", label, count)
    review = int(frame["needs_manual_review"].sum())
    if review:
        LOGGER.warning("%d campioni richiedono revisione manuale dell'istotipo", review)
    return frame


def histotype_classification_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Traceability table of the classification (one row per sample)."""
    columns = [
        "gsm_id",
        "title",
        "histotype_original",
        "histotype_normalized",
        "histotype_rule",
        "histotype_confidence",
        "needs_manual_review",
    ]
    available = [c for c in columns if c in frame.columns]
    return frame[available].copy()


def histotype_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count of samples per normalized histotype."""
    counts = (
        frame["histotype_normalized"]
        .value_counts(dropna=False)
        .rename_axis("histotype")
        .reset_index(name="n_samples")
    )
    total = int(counts["n_samples"].sum()) or 1
    counts["percent"] = (counts["n_samples"] / total * 100).round(2)
    return counts


# --------------------------------------------------------------------------- #
# Sample selection
# --------------------------------------------------------------------------- #


@dataclass
class SampleSets:
    """The three sample sets produced by the pipeline."""

    all_samples: pd.DataFrame
    non_serous: pd.DataFrame
    allowed: pd.DataFrame
    to_review: pd.DataFrame

    def summary(self) -> dict[str, int]:
        return {
            "n_all_samples": len(self.all_samples),
            "n_non_serous": len(self.non_serous),
            "n_allowed": len(self.allowed),
            "n_to_review": len(self.to_review),
        }


def build_sample_sets(
    frame: pd.DataFrame,
    allowed_histotypes: Sequence[str],
    excluded_histotypes: Sequence[str],
    keep_unspecified: bool = False,
) -> SampleSets:
    """Build the all / non-serous / allowed-histotype sets.

    The "non-serous" set is always defined by the ``Serous`` label, regardless of
    the configuration: it therefore stays available even when serous carcinomas
    are included among the allowed histotypes.

    ``Other or unspecified`` is NOT treated as equivalent to "non-serous": it
    belongs to the non-serous set but stays out of the final dataset unless
    ``keep_unspecified`` is ``True``.

    """
    excluded = {h.strip().lower() for h in excluded_histotypes}
    allowed = {h.strip().lower() for h in allowed_histotypes} - excluded

    labels = frame["histotype_normalized"].str.strip().str.lower()
    non_serous = frame[labels != SEROUS.lower()].copy()

    allowed_mask = labels.isin(allowed)
    if keep_unspecified:
        allowed_mask = allowed_mask | (labels == UNSPECIFIED.lower())
        LOGGER.warning(
            "keep_unspecified=True: i campioni '%s' sono inclusi nel dataset finale",
            UNSPECIFIED,
        )
    allowed_frame = frame[allowed_mask].copy()

    to_review = frame[frame["needs_manual_review"].fillna(False)].copy()

    LOGGER.info(
        "Insiemi di campioni: totali=%d, non sierosi=%d, istotipi ammessi=%d, "
        "da revisionare=%d",
        len(frame),
        len(non_serous),
        len(allowed_frame),
        len(to_review),
    )
    if allowed_frame.empty:
        LOGGER.error(
            "Nessun campione appartiene agli istotipi ammessi %s: "
            "verificare la lista in config.py o l'esito della classificazione.",
            list(allowed_histotypes),
        )
    return SampleSets(frame, non_serous, allowed_frame, to_review)


def filter_matrix_by_samples(
    matrix: pd.DataFrame, sample_columns: Iterable[str]
) -> pd.DataFrame:
    """Subset of matrix columns, preserving their order."""
    wanted = [c for c in matrix.columns if c in set(sample_columns)]
    return matrix[wanted].copy()
