#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch EasyRead metrics computation and analysis.

For each entry in training_data/metadata.json:
  - compute raw EasyRead metrics
  - compute EasyRead component scores and final EasyRead score
  - write them back into the metadata

Then:
  - print summary statistics of the scores
  - save plots under training_data/analysis/
"""
import sys
import os
import json
import csv
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Point to the repo root
EASYREAD_ROOT = Path("/work/courses/dslab/team4/easyread_project")

# Directory that contains easyread_metrics.py
EVAL_DIR = EASYREAD_ROOT / "src" / "evaluation"

# Make sure that directory is on sys.path
sys.path.insert(0, str(EVAL_DIR))

# Now this should work
from easyread_metrics import compute_metrics, compute_easyread_components_from_raw

DATA_ROOT = Path("/work/courses/dslab/team4/easyread_project/data").resolve()
TRAINING_DIR = DATA_ROOT / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
METADATA_JSON = TRAINING_DIR / "metadata.json"
METADATA_CSV = TRAINING_DIR / "metadata.csv"
ANALYSIS_DIR = TRAINING_DIR / "analysis"

BATCH_SIZE = 50


def atomic_json_dump(obj, path: Path) -> None:
    """
    Atomically write JSON to 'path' using a temporary file + os.replace.
    Ensures 'path' is always either the old valid file or the new valid file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_csv_write(rows, fieldnames, path: Path) -> None:
    """
    Atomically write CSV rows to 'path' using a temporary file + os.replace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_metadata():
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"metadata.json not found at {METADATA_JSON}")
    with METADATA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(all_metadata):
    """
    Save metadata.json and metadata.csv atomically.
    Killing the process in the middle of this function will not corrupt
    the existing JSON/CSV; only the temp files are affected.
    """
    # 1) JSON (atomic)
    atomic_json_dump(all_metadata, METADATA_JSON)

    # If there's no metadata, nothing else to do.
    if not all_metadata:
        return

    # 2) Determine CSV fieldnames
    fieldnames = set()
    for e in all_metadata:
        fieldnames.update(e.keys())

    preferred = [
        "dataset",
        "image_file",
        "id",
        "title",
        "keywords",
        "categories",
        "license",
        "prompt",
        "prompt_model",
        "easyread_palette_count_regions",
        "easyread_edge_density",
        "easyread_saliency_concentration",
        "easyread_delta_L",
        "easyread_relative_stroke_median",
        "easyread_centering_error",
        "easyread_palette_score",
        "easyread_edge_score",
        "easyread_saliency_score",
        "easyread_contrast_score",
        "easyread_stroke_score",
        "easyread_centering_score",
        "easyread_score",
    ]

    ordered = preferred + [k for k in sorted(fieldnames) if k not in preferred]

    # Prepare rows, flattening list fields
    csv_rows = []
    for e in all_metadata:
        row = e.copy()
        if isinstance(row.get("keywords"), list):
            row["keywords"] = "|".join(map(str, row["keywords"]))
        if isinstance(row.get("categories"), list):
            row["categories"] = "|".join(map(str, row["categories"]))
        csv_rows.append(row)

    # 3) CSV (atomic)
    atomic_csv_write(csv_rows, ordered, METADATA_CSV)


def analyze_scores(all_metadata):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    scores = []
    palette_vals = []
    datasets = []

    for e in all_metadata:
        score = e.get("easyread_score")
        if not isinstance(score, (int, float)):
            continue
        scores.append(float(score))
        palette_vals.append(float(e.get("easyread_palette_count_regions", np.nan)))
        ds = e.get("dataset") or "unknown"
        datasets.append(ds)

    if not scores:
        print("\n[ANALYSIS] No EasyRead scores found, nothing to analyze.")
        return

    scores_arr = np.array(scores, dtype=float)
    palette_arr = np.array(palette_vals, dtype=float)
    datasets_arr = np.array(datasets, dtype=object)

    print("\n[ANALYSIS] EasyRead score summary (all images)")
    print(f"  count: {scores_arr.size}")
    print(f"  mean:  {scores_arr.mean():.4f}")
    print(f"  std:   {scores_arr.std(ddof=1):.4f}")
    print(f"  min:   {scores_arr.min():.4f}")
    print(f"  max:   {scores_arr.max():.4f}")
    for q in [5, 25, 50, 75, 95]:
        v = np.percentile(scores_arr, q)
        print(f"  p{q:02d}:  {v:.4f}")

    # Per-dataset mean scores
    per_ds_scores = defaultdict(list)
    for ds, sc in zip(datasets_arr, scores_arr):
        per_ds_scores[ds].append(sc)

    print("\n[ANALYSIS] Mean EasyRead score per dataset:")
    ds_stats = []
    for ds, vals in per_ds_scores.items():
        arr = np.array(vals, dtype=float)
        ds_stats.append((ds, arr.size, float(arr.mean())))
    ds_stats.sort(key=lambda x: x[2], reverse=True)

    for ds, n, m in ds_stats:
        print(f"  {ds:20s}  n={n:5d}  mean={m:.4f}")

    # Histogram of scores
    plt.figure()
    plt.hist(scores_arr, bins=30)
    plt.xlabel("EasyRead score")
    plt.ylabel("Count")
    plt.title("Distribution of EasyRead scores")
    out_hist = ANALYSIS_DIR / "easyread_score_hist.png"
    plt.tight_layout()
    plt.savefig(str(out_hist))
    plt.close()
    print(f"\n[ANALYSIS] Saved histogram to {out_hist}")

    # Bar plot of mean score per dataset (only if more than one dataset)
    if len(ds_stats) > 1:
        ds_labels = [x[0] for x in ds_stats]
        ds_means = np.array([x[2] for x in ds_stats], dtype=float)

        plt.figure(figsize=(max(6, 0.4 * len(ds_labels)), 4))
        x = np.arange(len(ds_labels))
        plt.bar(x, ds_means)
        plt.xticks(x, ds_labels, rotation=45, ha="right")
        plt.ylabel("Mean EasyRead score")
        plt.title("Mean EasyRead score per dataset")
        plt.tight_layout()
        out_bar = ANALYSIS_DIR / "easyread_score_by_dataset.png"
        plt.savefig(str(out_bar))
        plt.close()
        print(f"[ANALYSIS] Saved per-dataset bar plot to {out_bar}")

    # Scatter: palette_count_regions vs score (filter NaNs)
    valid_mask = np.isfinite(palette_arr)
    if valid_mask.sum() > 0:
        plt.figure()
        plt.scatter(palette_arr[valid_mask], scores_arr[valid_mask], alpha=0.6)
        plt.xlabel("Palette count (regions)")
        plt.ylabel("EasyRead score")
        plt.title("EasyRead score vs palette complexity")
        plt.tight_layout()
        out_scatter = ANALYSIS_DIR / "easyread_palette_vs_score.png"
        plt.savefig(str(out_scatter))
        plt.close()
        print(f"[ANALYSIS] Saved palette vs score scatter to {out_scatter}")


def process_dataset():
    if not TRAINING_DIR.exists():
        raise FileNotFoundError(f"Training dir missing: {TRAINING_DIR}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images dir missing: {IMAGES_DIR}")
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"metadata.json missing: {METADATA_JSON}")

    metadata = load_metadata()
    total = len(metadata)
    updated = 0
    skipped_missing_img = 0
    skipped_existing = 0
    updated_since_last_save = 0

    for i, entry in enumerate(metadata, start=1):
        img_name = entry.get("image_file")
        if not img_name:
            continue

        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            skipped_missing_img += 1
            continue

        if "easyread_score" in entry:
            skipped_existing += 1
            continue

        print(f"[{i}/{total}] {img_name} -> computing EasyRead metrics...")

        try:
            raw = compute_metrics(str(img_path))
            components = compute_easyread_components_from_raw(raw)
        except Exception as e:
            print(f"  [WARN] Failed to compute metrics for {img_name}: {e}")
            continue

        entry["easyread_palette_count_regions"] = float(
            raw["palette"].get("palette_count_regions", 0.0)
        )
        entry["easyread_edge_density"] = float(
            raw["edges"].get("edge_density", 0.0)
        )
        entry["easyread_saliency_concentration"] = float(
            raw["saliency"].get("saliency_concentration", 0.0)
        )
        entry["easyread_delta_L"] = float(
            raw["contrast"].get("delta_L", 0.0)
        )
        entry["easyread_relative_stroke_median"] = float(
            raw["stroke"].get("relative_stroke_median", 0.0)
        )
        entry["easyread_centering_error"] = float(
            raw["centering_occupancy"].get("centering_error", 0.0)
        )

        entry["easyread_palette_score"] = float(components["palette_score"])
        entry["easyread_edge_score"] = float(components["edge_score"])
        entry["easyread_saliency_score"] = float(components["saliency_score"])
        entry["easyread_contrast_score"] = float(components["contrast_score"])
        entry["easyread_stroke_score"] = float(components["stroke_score"])
        entry["easyread_centering_score"] = float(components["centering_score"])

        entry["easyread_score"] = float(components["easyread_score"])

        updated += 1
        updated_since_last_save += 1

        if updated_since_last_save >= BATCH_SIZE:
            print(f"\n[INFO] Processed {updated_since_last_save} new entries, saving batch...")
            save_metadata(metadata)
            print(f"[INFO] Saved batch to:\n  {METADATA_JSON}\n  {METADATA_CSV}\n")
            updated_since_last_save = 0

    if updated_since_last_save > 0:
        print(f"\n[INFO] Final batch of {updated_since_last_save} entries, saving...")
        save_metadata(metadata)
        print(f"[INFO] Saved final batch to:\n  {METADATA_JSON}\n  {METADATA_CSV}\n")

    print("\n[SUMMARY]")
    print(f"  Total entries:           {total}")
    print(f"  Updated with EasyRead:   {updated}")
    print(f"  Skipped (missing image): {skipped_missing_img}")
    print(f"  Skipped (already had):   {skipped_existing}")

    analyze_scores(metadata)


if __name__ == "__main__":
    process_dataset()
