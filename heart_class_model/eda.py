"""
EDA visualizations for the PTB-XL preprocessing pipeline.

All figures are saved as high-DPI PNGs under <output_dir>/figures/eda/.
Depends only on matplotlib, numpy, pandas, tqdm, and wfdb (no torch/sklearn).

Exported entry point
--------------------
    run_eda(df, splits, output_dir, max_records=2000, resolution=100)

Called from preprocess_ptbxl.py when --eda is set. Can also be imported and
called directly for interactive exploration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for headless / CI runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Colorblind-friendly palette (one colour per superclass)
_PALETTE  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
_CLS_CLR  = dict(zip(SUPERCLASSES, _PALETTE))

# Colors for train / val / test
_SPLIT_CLR = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _eda_dir(output_dir: Path) -> Path:
    d = output_dir / "figures" / "eda"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path.name)


def _despine(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


# ── A. Class and split distributions ──────────────────────────────────────────

def plot_label_distributions(
    df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Four figures covering metadata-level distributions:

    01 — Global superclass prevalence (counts + percentages)
    02 — Per-split superclass prevalence (grouped bar)
    03 — Multi-label cardinality histogram
    04 — strat_fold record coverage bar chart
    """
    eda = _eda_dir(out_dir)

    # ── 01. Global label prevalence ───────────────────────────────────────────
    counts = df[SUPERCLASSES].sum()
    pcts   = 100.0 * df[SUPERCLASSES].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, ylabel, title, fmt in [
        (axes[0], counts, "Record count",       "Global label prevalence (n)",   "%.0f"),
        (axes[1], pcts,   "Positive records %", "Global label prevalence (%)",   "%.1f%%"),
    ]:
        bars = ax.bar(
            SUPERCLASSES, vals.values,
            color=_PALETTE, edgecolor="white", linewidth=0.8,
        )
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, vals.max() * 1.2)
        _despine(ax)

    fig.suptitle(
        f"PTB-XL  —  Superclass label prevalence  (N = {len(df):,})",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, eda / "01_global_label_prevalence.png")

    # ── 02. Per-split prevalence ───────────────────────────────────────────────
    n_classes  = len(SUPERCLASSES)
    n_splits   = len(splits)
    x          = np.arange(n_classes)
    bar_w      = 0.22

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (split_name, split_df) in enumerate(splits.items()):
        pct    = 100.0 * split_df[SUPERCLASSES].mean().values
        offset = (i - n_splits / 2 + 0.5) * bar_w
        bars   = ax.bar(
            x + offset, pct, bar_w,
            label=f"{split_name} (n={len(split_df):,})",
            color=_SPLIT_CLR[split_name],
            edgecolor="white", linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(SUPERCLASSES)
    ax.set_ylabel("Positive records (%)")
    ax.set_title("Label prevalence by split", fontweight="bold")
    ax.legend(title="Split")
    _despine(ax)
    fig.tight_layout()
    _save(fig, eda / "02_per_split_label_prevalence.png")

    # ── 03. Multi-label cardinality ────────────────────────────────────────────
    cardinality = df[SUPERCLASSES].sum(axis=1)
    vc          = cardinality.value_counts().sort_index()
    labels_str  = [str(int(k)) for k in vc.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels_str, vc.values, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of diagnostic labels per record")
    ax.set_ylabel("Record count")
    ax.set_title("Multi-label cardinality distribution", fontweight="bold")
    _despine(ax)
    fig.tight_layout()
    _save(fig, eda / "03_multilabel_cardinality.png")

    # ── 04. strat_fold coverage ────────────────────────────────────────────────
    fold_counts = df["strat_fold"].value_counts().sort_index()
    fold_colors = [
        "#55A868" if f <= 8 else "#DD8452" if f == 9 else "#C44E52"
        for f in fold_counts.index
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        fold_counts.index.astype(str), fold_counts.values,
        color=fold_colors, edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("strat_fold")
    ax.set_ylabel("Record count")
    ax.set_title("Record counts per stratification fold", fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#55A868", label="Train (folds 1–8)"),
            mpatches.Patch(facecolor="#DD8452", label="Val   (fold 9)"),
            mpatches.Patch(facecolor="#C44E52", label="Test  (fold 10)"),
        ],
        loc="upper right",
    )
    _despine(ax)
    fig.tight_layout()
    _save(fig, eda / "04_strat_fold_coverage.png")


# ── B. Co-occurrence heatmap ───────────────────────────────────────────────────

def plot_cooccurrence(df: pd.DataFrame, out_dir: Path) -> None:
    """
    5×5 superclass co-occurrence heatmap — raw counts and row-normalised (%).
    The diagonal is the per-class count; off-diagonal cells show how often two
    classes co-occur in the same record.
    """
    eda = _eda_dir(out_dir)

    label_mat = df[SUPERCLASSES].values.astype(int)   # (N, 5)
    cooc      = label_mat.T @ label_mat                # (5, 5)

    diag = cooc.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.where(diag[:, None] > 0, cooc / diag[:, None] * 100.0, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, data, title, is_pct in [
        (axes[0], cooc,     "Co-occurrence (counts)",          False),
        (axes[1], row_norm, "Row-normalised (% of row class)", True),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(SUPERCLASSES)
        ax.set_yticklabels(SUPERCLASSES)
        ax.set_xlabel("Column class")
        ax.set_ylabel("Row class")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.82)

        threshold = data.max() * 0.6
        for i in range(5):
            for j in range(5):
                val = data[i, j]
                txt = f"{val:.1f}%" if is_pct else f"{int(val):,}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center", fontsize=9,
                    color="white" if val > threshold else "black",
                )

    fig.suptitle("PTB-XL  —  Superclass co-occurrence", fontweight="bold")
    fig.tight_layout()
    _save(fig, eda / "05_label_cooccurrence.png")


# ── C. Box-and-whisker by class and lead ──────────────────────────────────────

def _load_lead_stats(
    df: pd.DataFrame,
    max_records: int,
    resolution: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load WFDB signals for up to *max_records* records and compute per-lead:
      - RMS   (root-mean-square amplitude)
      - PTP   (peak-to-peak amplitude)

    Uses a stratified random sample (proportional per strat_fold) so the
    subsampled distribution is representative of the full dataset.

    Returns
    -------
    stats_df   : DataFrame with ecg_id index, label cols, rms_<lead>, ptp_<lead>
    lead_names : list of lead name strings in WFDB channel order
    """
    import wfdb

    fp_col = f"filepath_{resolution}hz"

    if len(df) > max_records:
        frac = max_records / len(df)
        sample_df = (
            df.groupby("strat_fold", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=42))
        ).head(max_records)
        log.info(
            "Subsampled %d / %d records for lead stats (%d Hz)",
            len(sample_df), len(df), resolution,
        )
    else:
        sample_df = df.copy()

    rows: list[dict] = []
    lead_names: list[str] | None = None

    for ecg_id, row in tqdm(
        sample_df.iterrows(), total=len(sample_df), desc="Reading WFDB signals"
    ):
        fp = row.get(fp_col, "")
        if not fp:
            continue
        try:
            signal, fields = wfdb.rdsamp(str(fp))
            # signal shape: (n_samples, n_leads)

            if lead_names is None:
                lead_names = fields["sig_name"]

            sig_f = signal.astype(float)
            rms = np.sqrt(np.nanmean(sig_f ** 2, axis=0))   # (n_leads,)
            ptp = np.nanmax(sig_f, axis=0) - np.nanmin(sig_f, axis=0)

            rec: dict = {"ecg_id": ecg_id}
            for cls in SUPERCLASSES:
                rec[cls] = int(row[cls])
            for i, lname in enumerate(fields["sig_name"]):
                rec[f"rms_{lname}"] = rms[i]
                rec[f"ptp_{lname}"] = ptp[i]

            rows.append(rec)

        except Exception as exc:
            log.debug("Skipped ecg_id=%s : %s", ecg_id, exc)

    if not rows:
        log.warning("No WFDB records could be read; skipping lead-level plots")
        return pd.DataFrame(), []

    stats_df = pd.DataFrame(rows).set_index("ecg_id")
    log.info("Lead stats computed for %d records", len(stats_df))
    return stats_df, lead_names or []


def plot_lead_boxplots_by_class(
    df: pd.DataFrame,
    out_dir: Path,
    max_records: int = 2000,
    resolution: int = 100,
) -> None:
    """
    Box-and-whisker plots of per-lead RMS amplitude, grouped by diagnostic class.

    Produces:
      06_<cls>_lead_rms_boxplot.png (×5) — one per superclass, C=1 vs C=0,
                                           12 lead subplots each.
      11_summary_lead_rms_by_class.png  — all 5 classes side-by-side (C=1 only),
                                           12 lead subplots.
    """
    eda    = _eda_dir(out_dir)
    fp_col = f"filepath_{resolution}hz"

    plot_df = df[df[fp_col].astype(str).str.strip() != ""].copy()
    stats_df, lead_names = _load_lead_stats(plot_df, max_records, resolution)

    if stats_df.empty or not lead_names:
        return

    n_leads = len(lead_names)
    n_rows, n_cols = 3, 4   # 12 subplots arranged as 3×4 grid

    _bp_common = dict(
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
        widths=0.5,
    )

    # ── 06–10. Per-superclass: C=1 vs C=0 ─────────────────────────────────────
    for cls in SUPERCLASSES:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes_flat = axes.flatten()

        for i, lead in enumerate(lead_names):
            ax  = axes_flat[i]
            col = f"rms_{lead}"
            if col not in stats_df.columns:
                ax.set_visible(False)
                continue

            pos = stats_df.loc[stats_df[cls] == 1, col].dropna().values
            neg = stats_df.loc[stats_df[cls] == 0, col].dropna().values

            bp = ax.boxplot([neg, pos], labels=[f"{cls}=0", f"{cls}=1"], **_bp_common)
            bp["boxes"][0].set_facecolor("#C8D8F0")
            bp["boxes"][1].set_facecolor(_CLS_CLR[cls])

            ax.set_title(f"Lead {lead}", fontsize=9, fontweight="bold")
            ax.set_ylabel("RMS (mV)", fontsize=7)
            ax.tick_params(labelsize=7)
            _despine(ax)

            # Annotate sample sizes inside the plot
            ax.text(
                0.98, 0.97,
                f"n₀={len(neg)}  n₁={len(pos)}",
                transform=ax.transAxes,
                fontsize=6, ha="right", va="top", color="dimgray",
            )

        for j in range(n_leads, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Lead RMS — {cls} (positive vs negative)\n"
            f"({resolution} Hz, ≤{max_records} records, stratified sample)",
            fontweight="bold",
        )
        fig.tight_layout()
        _save(fig, eda / f"06_{cls.lower()}_lead_rms_boxplot.png")

    # ── 11. Summary: all classes, C=1 only ─────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes_flat = axes.flatten()

    for i, lead in enumerate(lead_names):
        ax  = axes_flat[i]
        col = f"rms_{lead}"
        if col not in stats_df.columns:
            ax.set_visible(False)
            continue

        class_data = [
            stats_df.loc[stats_df[cls] == 1, col].dropna().values
            for cls in SUPERCLASSES
        ]
        bp = ax.boxplot(class_data, labels=SUPERCLASSES, **_bp_common)
        for box, color in zip(bp["boxes"], _PALETTE):
            box.set_facecolor(color)

        ax.set_title(f"Lead {lead}", fontsize=9, fontweight="bold")
        ax.set_ylabel("RMS (mV)", fontsize=7)
        ax.tick_params(labelsize=7, axis="x", rotation=15)
        _despine(ax)

    for j in range(n_leads, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Lead RMS by superclass (positive-class records only)\n"
        f"({resolution} Hz, ≤{max_records} records, stratified sample)",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, eda / "11_summary_lead_rms_by_class.png")


# ── D. Demographics ────────────────────────────────────────────────────────────

def plot_demographics(
    df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Age histogram and sex bar chart per split, when those columns are present.
    PTB-XL encodes sex as 0 = male, 1 = female.
    Skips gracefully if columns are absent or too sparse.
    """
    eda = _eda_dir(out_dir)

    has_age = "age" in df.columns and df["age"].notna().sum() > 20
    has_sex = "sex" in df.columns and df["sex"].notna().sum() > 20

    if not has_age and not has_sex:
        log.info("No age / sex columns found — skipping demographics plot")
        return

    n_panels = int(has_age) + int(has_sex)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if has_age:
        ax = axes[panel]
        for split_name, split_df in splits.items():
            ages = split_df["age"].dropna()
            ax.hist(
                ages, bins=25, alpha=0.6,
                label=f"{split_name} (n={len(ages):,})",
                color=_SPLIT_CLR[split_name], edgecolor="white",
            )
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Count")
        ax.set_title("Age distribution by split", fontweight="bold")
        ax.legend()
        _despine(ax)
        panel += 1

    if has_sex:
        ax = axes[panel]
        split_names   = list(splits.keys())
        sex_map       = {0: "Male", 1: "Female"}
        x             = np.arange(len(split_names))
        bar_w         = 0.35
        sex_colors    = ["#4C72B0", "#C44E52"]

        for j, (sex_val, sex_label) in enumerate(sex_map.items()):
            counts = [(splits[sn]["sex"] == sex_val).sum() for sn in split_names]
            offset = (j - 0.5) * bar_w
            bars   = ax.bar(
                x + offset, counts, bar_w,
                label=sex_label, color=sex_colors[j], edgecolor="white",
            )
            ax.bar_label(bars, padding=2, fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(split_names)
        ax.set_ylabel("Count")
        ax.set_title("Sex distribution by split", fontweight="bold")
        ax.legend()
        _despine(ax)

    fig.suptitle("PTB-XL  —  Patient demographics", fontweight="bold")
    fig.tight_layout()
    _save(fig, eda / "12_demographics.png")


# ── E. Waveform sanity panel ───────────────────────────────────────────────────

def plot_example_waveforms(
    df: pd.DataFrame,
    out_dir: Path,
    resolution: int = 100,
) -> None:
    """
    12-lead waveform grid for one NORM and one abnormal (MI or STTC) example.
    Useful for verifying lead ordering and signal amplitude after path resolution.
    """
    import wfdb

    eda    = _eda_dir(out_dir)
    fp_col = f"filepath_{resolution}hz"

    candidates: list[tuple[str, pd.Series]] = []

    norm_rows = df[(df["NORM"] == 1) & (df[fp_col].astype(str).str.strip() != "")]
    if not norm_rows.empty:
        candidates.append(("NORM (normal sinus)", norm_rows.iloc[0]))

    abn_rows = df[
        ((df["MI"] == 1) | (df["STTC"] == 1))
        & (df["NORM"] == 0)
        & (df[fp_col].astype(str).str.strip() != "")
    ]
    if not abn_rows.empty:
        candidates.append(("Abnormal  MI / STTC", abn_rows.iloc[0]))

    for title, row in candidates:
        fp = str(row[fp_col])
        try:
            signal, fields = wfdb.rdsamp(fp)
        except Exception as exc:
            log.warning("Could not read waveform for '%s': %s", title, exc)
            continue

        n_samples, n_leads = signal.shape
        fs          = fields["fs"]
        t           = np.arange(n_samples) / fs
        lead_names  = fields["sig_name"]
        units       = fields.get("units", ["mV"] * n_leads)

        label_parts = [cls for cls in SUPERCLASSES if int(row[cls]) == 1]
        labels_str  = ", ".join(label_parts) if label_parts else "unlabeled"
        ecg_id      = row.name

        fig, axes = plt.subplots(
            n_leads, 1, figsize=(14, n_leads * 1.4), sharex=True
        )
        if n_leads == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t, signal[:, i], linewidth=0.7, color="#1f3e6d")
            ax.set_ylabel(
                f"{lead_names[i]}\n({units[i]})",
                fontsize=7.5, rotation=0, labelpad=48, va="center",
            )
            ax.tick_params(labelsize=7)
            _despine(ax)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"ECG id {ecg_id}  —  {title}\n"
            f"Labels: {labels_str}   |   {resolution} Hz  |  "
            f"Duration: {n_samples / fs:.1f} s",
            fontweight="bold",
        )
        fig.tight_layout()

        safe = (
            title.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_")
        )
        _save(fig, eda / f"13_waveform_{safe}.png")


# ── Master runner ──────────────────────────────────────────────────────────────

def run_eda(
    df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    max_records: int = 2000,
    resolution: int = 100,
) -> None:
    """
    Run the full EDA suite for the preprocessed PTB-XL dataset.

    Parameters
    ----------
    df          : full labeled DataFrame (output of build_label_columns + resolve_filepaths)
    splits      : dict with keys "train", "val", "test"
    output_dir  : pipeline output root; figures go to <output_dir>/figures/eda/
    max_records : max number of WFDB records to load for signal-derived plots
    resolution  : signal resolution in Hz (100 or 500) for lead-level plots
    """
    eda_dir = _eda_dir(output_dir)
    log.info("EDA output dir: %s", eda_dir)

    log.info("[EDA 1/5] Label distributions …")
    plot_label_distributions(df, splits, output_dir)

    log.info("[EDA 2/5] Co-occurrence heatmap …")
    plot_cooccurrence(df, output_dir)

    log.info(
        "[EDA 3/5] Lead box plots by class (max %d records, %d Hz) …",
        max_records, resolution,
    )
    plot_lead_boxplots_by_class(df, output_dir, max_records, resolution)

    log.info("[EDA 4/5] Demographics …")
    plot_demographics(df, splits, output_dir)

    log.info("[EDA 5/5] Waveform sanity panel …")
    plot_example_waveforms(df, output_dir, resolution)

    log.info("EDA complete. All figures in %s", eda_dir)
