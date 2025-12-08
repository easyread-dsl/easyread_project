#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Script (MERGE MODE)
Integrates multiple icon/pictogram datasets (arasaac, icon645, lds, openmoji)
into a unified training_data directory with standardized naming.

Key behaviors:
- Loads existing metadata.json (if present) and MERGES new entries into it.
- Deduplicates by (dataset, image_file).
- Avoids filename collisions in training_data/images via auto-renaming (_1, _2, ...).
- Writes metadata once at the end (with .bak backups of old files).
- NEW: Includes source subfolder names in output filenames to avoid collisions.
"""

import json
import shutil
from pathlib import Path
import csv
from typing import Dict, List, Any, Tuple
import os

# ------------------------- Base paths -------------------------
DATA_DIR = Path("/mnt/data2")
TRAINING_DIR = DATA_DIR / "training_data_arsaac"

# Dataset source directories
ARASAAC_DIR = DATA_DIR / "arasaac"
ICON645_DIR = DATA_DIR / "icon645"
LDS_DIR = DATA_DIR / "lds"
OPENMOJI_DIR = DATA_DIR / "openmoji"


# ------------------------- Helpers -------------------------
def move_preserve_meta(src: Path, dst: Path) -> None:
    """
    Move file from src to dst.
    If src and dst are on different filesystems, shutil.move falls back to copy2+unlink,
    preserving metadata. Raises on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def setup_training_directory():
    """Create training_data directory structure."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINING_DIR / "images").mkdir(exist_ok=True)
    print(f"[INFO] Training directory set up at: {TRAINING_DIR}")


def load_existing_metadata() -> List[Dict[str, Any]]:
    """Load existing metadata.json if present; otherwise return empty list."""
    json_path = TRAINING_DIR / "metadata.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def build_existing_keys(meta: List[Dict[str, Any]]) -> set:
    """Build a set of (dataset, image_file) for fast deduplication."""
    keys = set()
    for e in meta:
        ds = e.get("dataset")
        img = e.get("image_file")
        if ds and img:
            keys.add((ds, img))
    return keys


def unique_dest(path: Path) -> Path:
    """
    Return a non-colliding destination path by appending _1, _2, ... if needed.
    Ensures we never overwrite an existing file in images/.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    k = 1
    while True:
        cand = path.with_name(f"{stem}_{k}{suffix}")
        if not cand.exists():
            return cand
        k += 1


def _clean_part(s: str) -> str:
    """Sanitize a path component for filenames: keep alnum, '-', '_', '.'; replace others with '-'."""
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in s)


def subfolder_suffix(src_file: Path, root: Path, strip_first: str | None = None) -> str:
    """
    Build a suffix like '_animals_cats' from subfolders of src_file relative to root.
    - If strip_first is provided and matches the first relative component, drop it
      (e.g., 'images' or 'colored_icons_final').
    """
    try:
        rel = src_file.relative_to(root)
    except ValueError:
        return ""  # src not under root; no suffix
    parts = list(rel.parts[:-1])  # parents only, no filename
    if parts and strip_first and parts[0] == strip_first:
        parts = parts[1:]
    parts = [_clean_part(p) for p in parts if p and p != "."]
    return ("_" + "_".join(parts)) if parts else ""


# ------------------------- Dataset processors (MERGE-SAFE) -------------------------
def process_arasaac(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 1000
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process ARASAAC dataset and append only new entries (merge-safe).
    Returns (new_entries, moved_count).
    """
    print("\n[ARASAAC] Processing dataset...")

    metadata_file = ARASAAC_DIR / "metadata.json"
    if not metadata_file.exists():
        print(f"[WARN] ARASAAC metadata not found at {metadata_file}")
        return [], 0

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)  # expected mapping: pic_id -> meta dict

    processed_data: List[Dict[str, Any]] = []
    moved_count = 0

    for i, (pic_id, meta) in enumerate(metadata.items(), 1):
        image_file = meta.get("image_file") or meta.get("file_name") or meta.get("filename")
        if not image_file:
            continue

        source_path = ARASAAC_DIR / image_file
        if not source_path.exists():
            alt = ARASAAC_DIR / "images" / image_file
            if alt.exists():
                source_path = alt
            else:
                continue

        # Include subfolders relative to ARASAAC root, dropping leading 'images' if present
        sub_sfx = subfolder_suffix(source_path, ARASAAC_DIR, strip_first="images")
        stem, suffix = Path(image_file).stem, Path(image_file).suffix
        new_filename = f"arasaac{sub_sfx}_{_clean_part(stem)}{suffix}"

        dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_path.name
        key = ("arasaac", final_filename)

        # Skip if already present and the file exists
        if key in existing_keys and dest_path.exists():
            continue

        # Move file if needed
        if not dest_path.exists():
            try:
                move_preserve_meta(source_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {source_path} -> {dest_path}: {e}")
                continue

        # Append metadata if new
        if key not in existing_keys:
            keywords = meta.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split("|") if k.strip()]

            categories = meta.get("categories", [])
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split("|") if c.strip()]

            entry = {
                "dataset": "arasaac",
                "image_file": final_filename,
                "id": pic_id,
                "title": meta.get("title"),
                "keywords": keywords,
                "categories": categories,
                "license": meta.get("license", "CC BY-NC-SA 4.0"),
                "skin_color": meta.get("skin_color"),
                "hair_color": meta.get("hair_color"),
                "background_color": meta.get("background_color"),
            }
            processed_data.append(entry)
            existing_keys.add(key)

        if i % save_interval == 0:
            print(f"[ARASAAC] Checkpoint: {moved_count} files moved so far")

    print(f"[ARASAAC] New entries: {len(processed_data)} | Files moved: {moved_count}")
    return processed_data, moved_count


def process_icon645(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 1000
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process ICON645 dataset and append only new entries (merge-safe).
    Returns (new_entries, moved_count).
    """
    print("\n[ICON645] Processing dataset...")

    icons_dir = ICON645_DIR / "colored_icons_final"
    if not icons_dir.exists():
        print(f"[WARN] ICON645 directory not found at {icons_dir}")
        return [], 0

    processed_data: List[Dict[str, Any]] = []
    moved_count = 0
    i = 0

    for category_dir in icons_dir.iterdir():
        if not category_dir.is_dir():
            continue

        for image_file in category_dir.glob("*.png"):
            i += 1
            # Include subfolders relative to ICON645 root, dropping 'colored_icons_final'
            sub_sfx = subfolder_suffix(image_file, ICON645_DIR, strip_first="colored_icons_final")
            stem, suffix = image_file.stem, image_file.suffix
            new_filename = f"icon645{sub_sfx}_{_clean_part(stem)}{suffix}"

            dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
            final_filename = dest_path.name
            key = ("icon645", final_filename)

            if key in existing_keys and dest_path.exists():
                continue

            if not dest_path.exists():
                try:
                    move_preserve_meta(image_file, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
                    continue

            if key not in existing_keys:
                # category is encoded in filename already; still keep metadata fields
                category_name = category_dir.name
                icon_id = image_file.stem
                entry = {
                    "dataset": "icon645",
                    "image_file": final_filename,
                    "id": icon_id,
                    "title": category_name,
                    "keywords": [category_name],
                    "categories": [category_name],
                    "license": "CC BY-NC-SA 4.0"
                }
                processed_data.append(entry)
                existing_keys.add(key)

            if i % save_interval == 0:
                print(f"[ICON645] Checkpoint: {moved_count} files moved so far")

    print(f"[ICON645] New entries: {len(processed_data)} | Files moved: {moved_count}")
    return processed_data, moved_count


def process_lds(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 1000
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process LDS dataset and append only new entries (merge-safe).
    Returns (new_entries, moved_count).
    """
    print("\n[LDS] Processing dataset...")

    if not LDS_DIR.exists():
        print(f"[WARN] LDS directory not found at {LDS_DIR}")
        return [], 0

    processed_data: List[Dict[str, Any]] = []
    moved_count = 0

    for i, image_file in enumerate(LDS_DIR.glob("*.png"), 1):
        # No subfolders in LDS (flat), keep as before
        new_filename = f"lds_{_clean_part(image_file.stem)}{image_file.suffix}"
        dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_path.name
        key = ("lds", final_filename)

        if key in existing_keys and dest_path.exists():
            continue

        if not dest_path.exists():
            try:
                move_preserve_meta(image_file, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
                continue

        if key not in existing_keys:
            label = image_file.stem
            keywords = [kw.strip() for kw in label.replace("-", " ").replace("_", " ").split()]
            entry = {
                "dataset": "lds",
                "image_file": final_filename,
                "id": label,
                "title": label.replace("-", " ").replace("_", " "),
                "keywords": keywords,
                "categories": ["lds"],
                "license": "Learning Design Symbols"
            }
            processed_data.append(entry)
            existing_keys.add(key)

        if i % save_interval == 0:
            print(f"[LDS] Checkpoint: {moved_count} files moved so far")

    print(f"[LDS] New entries: {len(processed_data)} | Files moved: {moved_count}")
    return processed_data, moved_count


def process_openmoji(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 1000
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process OpenMoji dataset and append only new entries (merge-safe).
    Returns (new_entries, moved_count).
    """
    print("\n[OPENMOJI] Processing dataset...")

    metadata_file = OPENMOJI_DIR / "data" / "openmoji.json"
    if not metadata_file.exists():
        print(f"[WARN] OpenMoji metadata not found at {metadata_file}")
        return [], 0

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    # Prefer 618x618, fallback to 72x72
    images_dir = OPENMOJI_DIR / "color" / "618x618"
    if not images_dir.exists():
        images_dir = OPENMOJI_DIR / "color" / "72x72"
        if not images_dir.exists():
            print(f"[WARN] OpenMoji images directory not found")
            return [], 0

    meta_map: Dict[str, Dict[str, Any]] = {}
    for item in metadata_list:
        hexcode = item.get("hexcode")
        if hexcode:
            meta_map[hexcode] = item

    processed_data: List[Dict[str, Any]] = []
    moved_count = 0

    for i, image_file in enumerate(images_dir.glob("*.png"), 1):
        # OpenMoji images are flat inside images_dir; no meaningful subfolders here
        hexcode = image_file.stem
        new_filename = f"openmoji_{_clean_part(hexcode)}{image_file.suffix}"
        dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_path.name
        key = ("openmoji", final_filename)

        if key in existing_keys and dest_path.exists():
            continue

        if not dest_path.exists():
            try:
                move_preserve_meta(image_file, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {image_file} -> {dest_path}: {e}")
                continue

        if key not in existing_keys:
            meta = meta_map.get(hexcode, {})
            annotation = meta.get("annotation", hexcode)
            tags = meta.get("tags", "")
            openmoji_tags = meta.get("openmoji_tags", "")
            group = meta.get("group", "")
            subgroups = meta.get("subgroups", "")

            all_tags: List[str] = []
            if tags:
                all_tags += [t.strip() for t in tags.split(",") if t.strip()]
            if openmoji_tags:
                all_tags += [t.strip() for t in openmoji_tags.split(",") if t.strip()]

            categories: List[str] = []
            if group:
                categories.append(group)
            if subgroups:
                categories.append(subgroups)

            entry = {
                "dataset": "openmoji",
                "image_file": final_filename,
                "id": hexcode,
                "title": annotation,
                "keywords": all_tags if all_tags else [annotation],
                "categories": categories if categories else ["emoji"],
                "license": "CC BY-SA 4.0"
            }
            processed_data.append(entry)
            existing_keys.add(key)

        if i % save_interval == 0:
            print(f"[OPENMOJI] Checkpoint: {moved_count} files moved so far")

    print(f"[OPENMOJI] New entries: {len(processed_data)} | Files moved: {moved_count}")
    return processed_data, moved_count


# ------------------------- Metadata I/O -------------------------
def save_metadata_with_backup(all_metadata: List[Dict[str, Any]]):
    """Backup existing metadata.* then write fresh JSON and CSV from all_metadata."""
    json_path = TRAINING_DIR / "metadata.json"
    csv_path = TRAINING_DIR / "metadata.csv"

    # Backups
    if json_path.exists():
        shutil.copy2(json_path, json_path.with_suffix(".json.bak"))
        print(f"[INFO] Backup saved: {json_path.with_suffix('.json.bak').name}")
    if csv_path.exists():
        shutil.copy2(csv_path, csv_path.with_suffix(".csv.bak"))
        print(f"[INFO] Backup saved: {csv_path.with_suffix('.csv.bak').name}")

    # JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved metadata JSON to: {json_path}")

    # CSV
    if all_metadata:
        fieldnames = ["dataset", "image_file", "id", "title", "keywords", "categories", "license", "skin_color", "hair_color", "background_color"]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in all_metadata:
                row = entry.copy()
                if isinstance(row.get("keywords"), list):
                    row["keywords"] = "|".join(row["keywords"])
                if isinstance(row.get("categories"), list):
                    row["categories"] = "|".join(row["categories"])
                writer.writerow(row)
        print(f"[INFO] Saved metadata CSV to: {csv_path}")


def generate_statistics(all_metadata: List[Dict[str, Any]]):
    """Generate and print statistics about the combined dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    dataset_counts: Dict[str, int] = {}
    for entry in all_metadata:
        dataset = entry.get("dataset", "unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print(f"\nTotal images in metadata: {len(all_metadata):,}")
    print("\nBreakdown by dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset:15s}: {count:6,d} images")

    all_categories = set()
    for entry in all_metadata:
        cats = entry.get("categories", [])
        if isinstance(cats, list):
            all_categories.update(cats)

    print(f"\nTotal unique categories: {len(all_categories):,}")
    print("="*60 + "\n")


# ------------------------- Main -------------------------
def main():
    """Main function to orchestrate dataset preparation in MERGE mode."""
    print("=" * 60)
    print("PREPARING COMBINED DATASET (MERGE MODE)")
    print("=" * 60)

    setup_training_directory()

    # Load existing metadata and prepare dedup keys
    all_metadata: List[Dict[str, Any]] = load_existing_metadata()
    existing_keys = build_existing_keys(all_metadata)
    grand_total = 0

    

    # # ICON645
    # dataset_data, n = process_icon645(all_metadata, existing_keys, save_interval=1000)
    # all_metadata.extend(dataset_data)
    # grand_total += n

    # # LDS
    # dataset_data, n = process_lds(all_metadata, existing_keys, save_interval=1000)
    # all_metadata.extend(dataset_data)
    # grand_total += n

    # # OPENMOJI
    # dataset_data, n = process_openmoji(all_metadata, existing_keys, save_interval=1000)
    # all_metadata.extend(dataset_data)
    # grand_total += n
    
    # ARASAAC
    dataset_data, n = process_arasaac(all_metadata, existing_keys, save_interval=1000)
    all_metadata.extend(dataset_data)
    grand_total += n
    

    # Final save (with backups)
    save_metadata_with_backup(all_metadata)

    # Stats
    generate_statistics(all_metadata)

    print("=" * 60)
    print(f"[SUCCESS] Merge complete!")
    print(f"[INFO] Training data location: {TRAINING_DIR}")
    print(f"[INFO] Files moved this run: {grand_total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
