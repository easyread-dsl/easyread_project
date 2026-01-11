#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast summary for training_data focused on prompts (no filesystem checks):

- totals and per-dataset counts (now includes aac + mulberry automatically via metadata)
- licenses, categories (top N)
- prompt stats: how many entries have prompts, per-dataset breakdown,
  prompt length buckets, and prompt_model counts
- light sanity checks: duplicate image_file names, duplicate ids/filenames

NOTE: This version does NOT touch the image files at all (no exists(), no sizes),
so it should be very fast even for hundreds of thousands of entries.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# -------- Paths --------
DATA_DIR = (Path(__file__).resolve().parent / "../../data").resolve()
TRAINING_DIR = DATA_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
META_JSON = TRAINING_DIR / "metadata.json"


# -------- Helpers --------
def human(n: int) -> str:
    return f"{n:,}"


def top_n(counter: Counter, n: int = 20) -> List[Tuple[str, int]]:
    return counter.most_common(n)


def bucket_prompt_length(length: int) -> str:
    """Bucket prompt character lengths for coarse stats."""
    if length == 0:
        return "0"
    if length <= 30:
        return "1–30"
    if length <= 60:
        return "31–60"
    if length <= 100:
        return "61–100"
    if length <= 200:
        return "101–200"
    return "200+"


# -------- Main summary --------
def main():
    if not META_JSON.exists():
        print(f"[ERROR] metadata.json not found at {META_JSON}")
        return

    # We don't actually need IMAGES_DIR to exist for this summary,
    # but we keep this check as a sanity note.
    if not IMAGES_DIR.exists():
        print(f"[WARN] images directory not found at {IMAGES_DIR} (not used in fast mode)")

    with META_JSON.open("r", encoding="utf-8") as f:
        meta: List[Dict] = json.load(f)

    total_meta = len(meta)
    print("=" * 70)
    print("TRAINING DATA SUMMARY (SUPER-FAST, PROMPT-FOCUSED, NO FS I/O)")
    print("=" * 70)
    print(f"[INFO] Metadata entries: {human(total_meta)}")
    print(f"[INFO] Images dir (for reference only): {IMAGES_DIR}")

    # Counters
    by_dataset = Counter()
    by_license = Counter()
    categories_counter = Counter()

    # Prompt-related counters
    prompts_present = 0
    prompts_missing = 0
    prompts_by_dataset = Counter()
    no_prompt_by_dataset = Counter()
    prompt_length_buckets = Counter()
    prompt_model_counter = Counter()

    # Sanity / diagnostics (metadata-only)
    duplicate_filenames = Counter()
    duplicate_ids = Counter()
    seen_filenames = set()
    seen_ids_per_dataset: Dict[str, set] = defaultdict(set)

    for e in meta:
        ds = e.get("dataset", "unknown")
        img = e.get("image_file")
        lic = e.get("license", "unknown")
        cats = e.get("categories", [])
        iid = e.get("id")

        by_dataset[ds] += 1
        by_license[lic] += 1
        if isinstance(cats, list):
            categories_counter.update(cats)

        # ----- prompt stats -----
        prompt = e.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            prompts_present += 1
            prompts_by_dataset[ds] += 1
            length = len(prompt.strip())
            prompt_length_buckets[bucket_prompt_length(length)] += 1

            pmodel = e.get("prompt_model")
            if isinstance(pmodel, str) and pmodel.strip():
                prompt_model_counter[pmodel.strip()] += 1
        else:
            prompts_missing += 1
            no_prompt_by_dataset[ds] += 1

        # ----- metadata-only duplicate checks -----
        if img is not None:
            # Check duplicate filenames based purely on the 'image_file' string
            if img in seen_filenames:
                duplicate_filenames[img] += 1
            else:
                seen_filenames.add(img)

        if iid is not None:
            if iid in seen_ids_per_dataset[ds]:
                duplicate_ids[(ds, str(iid))] += 1
            else:
                seen_ids_per_dataset[ds].add(iid)

    # -------- Print summary --------
    print("\n--- Totals ---")
    print(f"Total entries in metadata: {human(total_meta)}")
    print(f"Entries with prompts:      {human(prompts_present)}")
    print(f"Entries without prompts:   {human(prompts_missing)}")
    if total_meta > 0:
        pct = (prompts_present / total_meta) * 100.0
        print(f"Share with prompts:        {pct:.2f}%")

    print("\n--- By dataset (entries) ---")
    for ds, cnt in sorted(by_dataset.items()):
        print(f"{ds:15s}: {human(cnt)}")

    print("\n--- Licenses ---")
    for lic, cnt in sorted(by_license.items(), key=lambda x: (-x[1], x[0])):
        print(f"{lic}: {human(cnt)}")

    print("\n--- Top categories (up to 20) ---")
    for cat, cnt in top_n(categories_counter, n=20):
        print(f"{cat}: {human(cnt)}")

    # -------- Prompt statistics --------
    print("\n--- Prompts by dataset ---")
    all_ds = sorted(by_dataset.keys())
    for ds in all_ds:
        with_p = prompts_by_dataset.get(ds, 0)
        without_p = no_prompt_by_dataset.get(ds, 0)
        total_ds = by_dataset.get(ds, 0)
        if total_ds > 0:
            pct_ds = (with_p / total_ds) * 100.0
        else:
            pct_ds = 0.0
        print(
            f"{ds:15s}: with={human(with_p):>6s}, "
            f"without={human(without_p):>6s}, "
            f"total={human(total_ds):>6s}, "
            f"{pct_ds:6.2f}% with"
        )

    print("\nPrompt length buckets (characters):")
    for bucket, cnt in sorted(prompt_length_buckets.items(), key=lambda x: x[0]):
        print(f"{bucket:>7s}: {human(cnt)}")

    if prompt_model_counter:
        print("\nPrompt models (how prompts were generated):")
        for model_name, cnt in top_n(prompt_model_counter, n=len(prompt_model_counter)):
            print(f"{model_name}: {human(cnt)}")

    # Sanity / diagnostics (metadata-only duplicates)
    if duplicate_filenames:
        print("\n[INFO] Duplicate image_file names in metadata (same string appears multiple times):")
        for fn, extra in top_n(duplicate_filenames, n=10):
            print(f"  {fn}: +{extra}")

    if duplicate_ids:
        print("\n[INFO] Duplicate IDs within a dataset in metadata (same (dataset,id) appears multiple times):")
        shown = 0
        for (ds, iid), extra in duplicate_ids.items():
            print(f"  ({ds}, id={iid}): +{extra}")
            shown += 1
            if shown >= 10:
                remaining = len(duplicate_ids) - shown
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

    print("\n" + "=" * 70)
    print("[DONE] Super-fast prompt-focused summary complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
