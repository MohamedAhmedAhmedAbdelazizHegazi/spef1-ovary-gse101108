"""Matching between expression-matrix columns and GEO samples.

The matching proceeds by decreasing reliability levels (exact, normalized,
by token, partial). A pairing is accepted only if it is unique in both
directions: ambiguities are not resolved arbitrarily but flagged for manual
review.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Metadata columns used as possible sample keys.
KEY_COLUMNS = ("gsm_id", "title", "source_name", "description", "sample_id")

#: Frequent prefixes/suffixes to remove from GEO titles.
AFFIXES = (
    "tumor", "tumour", "sample", "patient", "case", "rnaseq", "rna", "seq",
    "ov", "ovary", "ovarian",
)

METHOD_SCORES = {
    "exact": 1.00,
    "normalized": 0.95,
    "token": 0.85,
    "partial": 0.60,
}


@dataclass
class MatchingResult:
    """Overall outcome of the matching."""

    table: pd.DataFrame
    matched: dict[str, str]
    unmatched_samples: list[str]
    unmatched_columns: list[str]
    conflicts: pd.DataFrame

    @property
    def match_fraction(self) -> float:
        total = len(self.table)
        return len(self.matched) / total if total else 0.0


def normalize_label(text: object) -> str:
    """Normalize a label: lower case, without non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def tokenize_label(text: object) -> set[str]:
    """Scompone un'etichetta in token normalizzati e informativi.

    ``"Tumor_OV106"`` produce ``{"tumorov106", "ov106", "106"}``.
    """
    raw = str(text or "").strip()
    if not raw:
        return set()
    pieces = [p for p in re.split(r"[\s\-_\.\|/:,]+", raw) if p]
    tokens: set[str] = {normalize_label(raw)}
    for piece in pieces:
        norm = normalize_label(piece)
        if norm:
            tokens.add(norm)
    # concatenazioni progressive: "tumor","ov106" -> "tumorov106"
    for size in range(2, len(pieces) + 1):
        for start in range(0, len(pieces) - size + 1):
            tokens.add(normalize_label("".join(pieces[start: start + size])))
    # removal of generic affixes
    for token in list(tokens):
        for affix in AFFIXES:
            if token.startswith(affix) and len(token) > len(affix) + 1:
                tokens.add(token[len(affix):])
            if token.endswith(affix) and len(token) > len(affix) + 1:
                tokens.add(token[: -len(affix)])
    return {t for t in tokens if len(t) >= 2}


def _sample_keys(row: pd.Series) -> tuple[list[str], set[str]]:
    """Raw keys and tokens of a sample."""
    raw_keys = [
        str(row[col]) for col in KEY_COLUMNS if col in row.index and pd.notna(row[col])
    ]
    tokens: set[str] = set()
    for key in raw_keys:
        tokens |= tokenize_label(key)
    return raw_keys, tokens


def match_samples_to_columns(
    metadata: pd.DataFrame, columns: Sequence[str]
) -> MatchingResult:
    """Match each GEO sample to a matrix column.

    Args:
        metadata: normalized metadata (must contain ``gsm_id``).
        columns: names of the matrix sample columns.

    Returns:
        :class:`MatchingResult` with a detailed table, a ``gsm_id -> column``
        dictionary, unmatched samples/columns and conflicts.

    """
    columns = [str(c) for c in columns]
    column_norm = {c: normalize_label(c) for c in columns}
    column_tokens = {c: tokenize_label(c) for c in columns}

    records: dict[str, dict] = {}
    for _, row in metadata.iterrows():
        gsm_id = str(row["gsm_id"])
        raw_keys, tokens = _sample_keys(row)
        records[gsm_id] = {
            "gsm_id": gsm_id,
            "title": row.get("title", ""),
            "histotype_normalized": row.get("histotype_normalized", ""),
            "raw_keys": raw_keys,
            "norm_keys": {normalize_label(k) for k in raw_keys},
            "tokens": tokens,
            "matrix_column": None,
            "match_method": "",
            "match_score": 0.0,
            "candidate_columns": "",
            "conflict": False,
        }

    assigned_columns: set[str] = set()

    for method in ("exact", "normalized", "token", "partial"):
        proposals: dict[str, list[str]] = {}
        for gsm_id, record in records.items():
            if record["matrix_column"] is not None:
                continue
            hits = [
                column
                for column in columns
                if column not in assigned_columns
                and _is_match(method, column, column_norm[column], column_tokens[column], record)
            ]
            if hits:
                proposals[gsm_id] = hits

        # inversion: a column proposed by several samples is ambiguous
        column_owners: dict[str, list[str]] = {}
        for gsm_id, hits in proposals.items():
            for column in hits:
                column_owners.setdefault(column, []).append(gsm_id)

        for gsm_id, hits in proposals.items():
            record = records[gsm_id]
            record["candidate_columns"] = ", ".join(hits)
            if len(hits) > 1:
                record["conflict"] = True
                LOGGER.warning(
                    "Campione %s: %d colonne candidate (%s) con metodo '%s': "
                    "nessuna assegnazione automatica.",
                    gsm_id,
                    len(hits),
                    ", ".join(hits[:5]),
                    method,
                )
                continue
            column = hits[0]
            if len(column_owners.get(column, [])) > 1:
                record["conflict"] = True
                LOGGER.warning(
                    "Colonna '%s' rivendicata da piu' campioni (%s): conflitto "
                    "non risolto automaticamente.",
                    column,
                    ", ".join(column_owners[column]),
                )
                continue
            record["matrix_column"] = column
            record["match_method"] = method
            record["match_score"] = METHOD_SCORES[method]
            record["conflict"] = False
            assigned_columns.add(column)

    table = pd.DataFrame(
        [
            {
                "gsm_id": r["gsm_id"],
                "title": r["title"],
                "histotype": r["histotype_normalized"],
                "matrix_column": r["matrix_column"],
                "match_method": r["match_method"] or "none",
                "match_score": r["match_score"],
                "candidate_columns": r["candidate_columns"],
                "conflict": r["conflict"],
                "matched": r["matrix_column"] is not None,
            }
            for r in records.values()
        ]
    )

    matched = {
        r["gsm_id"]: r["matrix_column"]
        for r in records.values()
        if r["matrix_column"] is not None
    }
    unmatched_samples = [r["gsm_id"] for r in records.values() if r["matrix_column"] is None]
    unmatched_columns = [c for c in columns if c not in assigned_columns]
    conflicts = table[table["conflict"]].copy()

    LOGGER.info(
        "Matching completato: %d/%d campioni associati (%.1f%%); "
        "%d colonne della matrice non associate",
        len(matched),
        len(records),
        100 * len(matched) / max(len(records), 1),
        len(unmatched_columns),
    )
    by_method = table.loc[table["matched"], "match_method"].value_counts().to_dict()
    LOGGER.info("Metodi di matching usati: %s", by_method or "nessuno")
    if unmatched_samples:
        LOGGER.warning(
            "Campioni senza colonna corrispondente: %s",
            ", ".join(unmatched_samples[:20]),
        )
    if unmatched_columns:
        LOGGER.warning(
            "Colonne della matrice senza campione GEO: %s",
            ", ".join(unmatched_columns[:20]),
        )

    return MatchingResult(table, matched, unmatched_samples, unmatched_columns, conflicts)


def _is_match(
    method: str, column: str, column_norm: str, column_tokens: set[str], record: dict
) -> bool:
    """Check the column/sample correspondence at the indicated level."""
    if method == "exact":
        return column in record["raw_keys"]
    if method == "normalized":
        return column_norm in record["norm_keys"]
    if method == "token":
        return bool(column_tokens & record["tokens"])
    if method == "partial":
        if len(column_norm) < 3:
            return False
        return any(
            column_norm in key or key in column_norm
            for key in record["norm_keys"]
            if len(key) >= 3
        )
    return False


def check_match_coverage(
    matching: MatchingResult,
    allowed_gsm_ids: Iterable[str],
    min_fraction: float,
    force: bool = False,
) -> tuple[bool, float, str]:
    """Check that a sufficient share of allowed samples is matched.

    Returns:
        ``(ok, fraction, message)``. With ``force=True`` ``ok`` is always ``True``
        but the message still reports the problem.

    """
    allowed = list(allowed_gsm_ids)
    if not allowed:
        return False, 0.0, "Nessun campione ammesso da associare."
    matched = sum(1 for gsm in allowed if gsm in matching.matched)
    fraction = matched / len(allowed)
    message = (
        f"{matched}/{len(allowed)} campioni ammessi associati alla matrice "
        f"({fraction:.1%}; soglia minima {min_fraction:.0%})"
    )
    if fraction >= min_fraction:
        return True, fraction, message
    message += (
        ". Il dataset finale non verra' creato: controllare "
        "data/processed/*_sample_matching.xlsx e, se il matching e' corretto, "
        "rilanciare con --force."
    )
    return (True if force else False), fraction, message
