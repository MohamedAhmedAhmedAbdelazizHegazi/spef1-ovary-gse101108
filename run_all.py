"""End-to-end driver that reproduces every dataset result of the manuscript.

The three stages are run in order:

    1. GSE101108_pipeline  - download GSE101108 from GEO, normalize the metadata,
                             classify the histotypes and build the expression
                             matrices used downstream.
    2. SPEF1_analysis      - SPEF1 co-expression, differential expression, GO
                             over-representation and immune-signature analyses;
                             writes the citable numbers and the base figures
                             (volcano, immune) and the results workbook.
    3. figure scripts      - the manuscript-ready figures and tables: the
                             histotype/clinical panel with the HGSC/LGSC split
                             (and the dataset/TMA tables), the co-expression
                             correlation matrix and clustering heatmap, and the
                             dataset GO enrichment in STRING style.

Run from the repository root::

    python run_all.py
    python run_all.py --skip-download     # reuse a previous GEO download
    python run_all.py --skip-tma          # skip the tissue-microarray table

The STRING network and enrichment figures (manuscript Figs 1-2) are not produced
here: they were exported directly from the string-db.org web interface (see README).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "GSE101108_pipeline"
ANALYSIS = ROOT / "SPEF1_analysis"


def run(cmd: list[str], cwd: Path) -> None:
    """Run one stage and stop the whole driver if it fails."""
    print(f"\n>>> {' '.join(cmd)}  (in {cwd.name})", flush=True)
    result = subprocess.run([sys.executable, *cmd], cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"stage failed ({cwd.name}/{cmd[0]}), exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the GSE101108 dataset results of the SPEF1 paper.")
    parser.add_argument("--skip-download", action="store_true", help="reuse a previous GEO download")
    parser.add_argument("--skip-enrichment", action="store_true", help="skip the local GO over-representation step")
    parser.add_argument("--skip-tma", action="store_true", help="skip the tissue-microarray table")
    args = parser.parse_args()

    # 1. data preparation
    pipe_cmd = ["main.py"]
    if args.skip_download:
        pipe_cmd.append("--skip-download")
    run(pipe_cmd, PIPE)

    # 2. core SPEF1 analysis (co-expression, differential, GO, immune; workbook + numbers)
    analysis_cmd = ["main.py"]
    if args.skip_enrichment:
        analysis_cmd.append("--skip-enrichment")
    run(analysis_cmd, ANALYSIS)

    # 3. manuscript figures and tables
    run(["regen_histotype_tables.py"], ANALYSIS)    # Fig 3 + Table 2 (+ Table 6 if TMA present)
    run(["regen_coexpression_figures.py"], ANALYSIS)  # Fig 4 correlation matrix + Fig 5 heatmap
    run(["regen_dataset_go.py"], ANALYSIS)          # Fig 7 dataset GO (STRING style)

    print("\nAll stages completed. Figures and tables are under SPEF1_analysis/results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
