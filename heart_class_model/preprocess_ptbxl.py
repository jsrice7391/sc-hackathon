#!/usr/bin/env python3
"""
PTB-XL ECG dataset preprocessing pipeline.

Converts raw downloaded PTB-XL files into structured CSV datasets ready for
feature extraction and model training.

Usage (from heart_class_model/):
    uv run python preprocess_ptbxl.py --data-dir /path/to/ptb-xl
    uv run python preprocess_ptbxl.py --data-dir /path/to/ptb-xl --drop-no-labels
    uv run python preprocess_ptbxl.py --data-dir /path/to/ptb-xl --eda
    uv run python preprocess_ptbxl.py --data-dir /path/to/ptb-xl --eda --eda-resolution 500

Output (default: heart_class_model/processed/):
    processed/
      100hz/train.csv, val.csv, test.csv
      500hz/train.csv, val.csv, test.csv
      figures/eda/*.png          (only when --eda is set)

CSV schema per split:
    ecg_id, patient_id, filepath, strat_fold, NORM, MI, STTC, CD, HYP

Split assignment (PTB-XL strat_fold column):
    folds 1–8 → train  |  fold 9 → val  |  fold 10 → test
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
SUPERCLASSES: list[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

log = logging.getLogger(__name__)


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )


# ── Step 1: Load metadata ──────────────────────────────────────────────────────

def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ptbxl_database.csv and scp_statements.csv.

    Returns
    -------
    df       : record-level DataFrame indexed by ecg_id; scp_codes parsed to dict
    diag_scp : scp_statements rows where diagnostic == 1.0
    """
    db_path  = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    for p in (db_path, scp_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Make sure --data-dir points to the PTB-XL root."
            )

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df   = pd.read_csv(scp_path, index_col=0)
    diag_scp = scp_df[scp_df["diagnostic"] == 1.0].copy()

    log.info("Loaded %d records from ptbxl_database.csv", len(df))
    log.info("Using %d diagnostic SCP codes from scp_statements.csv", len(diag_scp))
    return df, diag_scp


# ── Step 2: Map SCP codes → five superclasses ──────────────────────────────────

def map_to_superclasses(
    scp_codes: dict, diag_scp: pd.DataFrame
) -> dict[str, int]:
    """
    Return {superclass: 1} for every diagnostic superclass present in one record.

    Parameters
    ----------
    scp_codes : dict from the scp_codes column of ptbxl_database.csv
    diag_scp  : scp_statements filtered to diagnostic == 1.0
    """
    result: dict[str, int] = {}
    for code in scp_codes:
        if code in diag_scp.index:
            cls = diag_scp.at[code, "diagnostic_class"]
            if pd.notna(cls) and cls in SUPERCLASSES:
                result[cls] = 1
    return result


def build_label_columns(df: pd.DataFrame, diag_scp: pd.DataFrame) -> pd.DataFrame:
    """Add binary NORM / MI / STTC / CD / HYP columns to df (in-place)."""
    mapped = df["scp_codes"].apply(lambda c: map_to_superclasses(c, diag_scp))
    label_df = (
        pd.DataFrame(mapped.tolist(), index=df.index)
        .reindex(columns=SUPERCLASSES, fill_value=0)
    )
    for col in SUPERCLASSES:
        df[col] = label_df[col].fillna(0).astype(int)

    n_labeled = df[SUPERCLASSES].any(axis=1).sum()
    log.info(
        "%d / %d records have at least one diagnostic label",
        n_labeled, len(df),
    )
    return df


# ── Step 3: Resolve file paths ─────────────────────────────────────────────────

def resolve_filepaths(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Build absolute WFDB stem paths for 100 Hz (filename_lr) and 500 Hz
    (filename_hr) recordings.  Logs missing .hea headers without dropping rows
    so the caller can decide how to handle them.
    """
    missing: dict[int, int] = {100: 0, 500: 0}
    lr_paths: list[str] = []
    hr_paths: list[str] = []

    for ecg_id, row in df.iterrows():
        lr_stem = data_dir / row["filename_lr"]
        hr_stem = data_dir / row["filename_hr"]

        if not lr_stem.with_suffix(".hea").exists():
            log.debug("Missing 100 Hz header  ecg_id=%s  %s", ecg_id, lr_stem)
            missing[100] += 1

        if not hr_stem.with_suffix(".hea").exists():
            log.debug("Missing 500 Hz header  ecg_id=%s  %s", ecg_id, hr_stem)
            missing[500] += 1

        lr_paths.append(str(lr_stem))
        hr_paths.append(str(hr_stem))

    df["filepath_100hz"] = lr_paths
    df["filepath_500hz"] = hr_paths

    for hz, n in missing.items():
        if n:
            log.warning("%d records are missing a %d Hz .hea header", n, hz)
        else:
            log.info("All %d Hz headers present", hz)

    return df


# ── Step 4: Train / val / test split ───────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df by strat_fold.

    Assignments
    -----------
    folds 1–8 → train | fold 9 → val | fold 10 → test

    Logs a warning if any patient_id appears in more than one split.
    """
    train = df[df["strat_fold"].between(1, 8)].copy()
    val   = df[df["strat_fold"] == 9].copy()
    test  = df[df["strat_fold"] == 10].copy()

    pid_sets = {
        "train": set(train["patient_id"]),
        "val":   set(val["patient_id"]),
        "test":  set(test["patient_id"]),
    }
    leakage_found = False
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = pid_sets[a] & pid_sets[b]
        if overlap:
            log.warning(
                "patient_id overlap %s / %s: %d shared patients",
                a, b, len(overlap),
            )
            leakage_found = True

    if not leakage_found:
        log.info("No patient_id leakage across splits (verified)")

    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ── Step 5: Save CSVs ──────────────────────────────────────────────────────────

def save_csvs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    resolution: int,
) -> None:
    """
    Write train / val / test CSV files for one resolution (100 or 500 Hz).

    Output path: <output_dir>/<resolution>hz/{train,val,test}.csv
    """
    hz_dir  = output_dir / f"{resolution}hz"
    hz_dir.mkdir(parents=True, exist_ok=True)
    fp_col  = f"filepath_{resolution}hz"

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = (
            split_df
            .reset_index()[["ecg_id", "patient_id", fp_col, "strat_fold"] + SUPERCLASSES]
            .rename(columns={fp_col: "filepath"})
        )
        out.to_csv(hz_dir / f"{name}.csv", index=False)
        log.info("  %dhz/%s.csv  → %d records", resolution, name, len(out))


# ── Step 6: Print dataset statistics ──────────────────────────────────────────

def print_stats(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Print split sizes, label distribution, and multi-label cardinality."""
    splits = {"train": train, "val": val, "test": test}
    sep    = "─" * 70

    # Record counts + raw positive counts per class
    header = f"{'Split':<8} {'Records':>8}  " + "  ".join(f"{c:>6}" for c in SUPERCLASSES)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, df in splits.items():
        row = f"{name:<8} {len(df):>8}  " + "  ".join(
            f"{df[c].sum():>6}" for c in SUPERCLASSES
        )
        print(row)
    print(sep)

    # Prevalence percentages
    print(f"\n{'Label prevalence (% positive per split)'}")
    hdr2 = f"{'Split':<8}  " + "  ".join(f"{c:>7}" for c in SUPERCLASSES)
    print(hdr2)
    print("─" * 52)
    for name, df in splits.items():
        pcts = "  ".join(f"{100 * df[c].mean():>6.1f}%" for c in SUPERCLASSES)
        print(f"{name:<8}  {pcts}")

    # Multi-label cardinality
    all_df = pd.concat(splits.values())
    card   = all_df[SUPERCLASSES].sum(axis=1).value_counts().sort_index()
    print("\nMulti-label cardinality (labels per record):")
    for n_labels, count in card.items():
        pct = 100 * count / len(all_df)
        print(f"  {int(n_labels)} label(s): {count:>6} records  ({pct:.1f}%)")
    print()


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess PTB-XL ECG dataset → structured CSV datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  uv run python preprocess_ptbxl.py --data-dir /data/ptb-xl --eda\n"
            "  uv run preprocess-ptbxl --data-dir /data/ptb-xl --drop-no-labels"
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="PTB-XL root directory (must contain ptbxl_database.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output root. Defaults to <script_dir>/processed/",
    )
    parser.add_argument(
        "--drop-no-labels",
        action="store_true",
        help="Drop records with no diagnostic superclass assignment",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run EDA visualizations; save PNGs to <output-dir>/figures/eda/",
    )
    parser.add_argument(
        "--eda-max-records",
        type=int,
        default=2000,
        metavar="N",
        help="Max records to load for signal-derived EDA (WFDB reads are slow)",
    )
    parser.add_argument(
        "--eda-resolution",
        type=int,
        choices=[100, 500],
        default=100,
        metavar="HZ",
        help="Signal resolution to use for lead-level EDA box plots",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    data_dir   = args.data_dir.resolve()
    output_dir = (
        Path(__file__).parent / "processed"
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("PTB-XL data dir : %s", data_dir)
    log.info("Output dir      : %s", output_dir)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    df, diag_scp = load_metadata(data_dir)
    df = build_label_columns(df, diag_scp)

    if args.drop_no_labels:
        n_before = len(df)
        df = df[df[SUPERCLASSES].any(axis=1)].copy()
        log.info(
            "Dropped %d records with no diagnostic labels (%d remain)",
            n_before - len(df), len(df),
        )

    df = resolve_filepaths(df, data_dir)
    train, val, test = split_dataset(df)

    log.info("Saving CSVs …")
    for res in (100, 500):
        save_csvs(train, val, test, output_dir, res)

    print_stats(train, val, test)

    if args.eda:
        from eda import run_eda  # local import; keeps non-EDA runs lightweight
        run_eda(
            df,
            splits={"train": train, "val": val, "test": test},
            output_dir=output_dir,
            max_records=args.eda_max_records,
            resolution=args.eda_resolution,
        )

    log.info("Pipeline complete. All outputs in %s", output_dir)


if __name__ == "__main__":
    main()
