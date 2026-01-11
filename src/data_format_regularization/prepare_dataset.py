#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preparation Script (MERGE MODE)
Integrates multiple icon/pictogram datasets (arasaac, aac, mulberry, icon645, lds, openmoji)
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
import subprocess
from pathlib import Path
import csv
from typing import Dict, List, Any, Tuple
import os

# ------------------------- Base paths -------------------------
DATA_DIR = Path("/mnt/data2")
TRAINING_DIR = DATA_DIR / "training_data"

# Dataset source directories
ARASAAC_DIR = DATA_DIR / "arasaac"
AAC_DIR = DATA_DIR / "aac"
MULBERRY_DIR = DATA_DIR / "mulberry"
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
    - If strip_first is provided and matches the first relative component, drop it.
    """
    try:
        rel = src_file.relative_to(root)
    except ValueError:
        return ""
    parts = list(rel.parts[:-1])
    if parts and strip_first and parts[0] == strip_first:
        parts = parts[1:]
    parts = [_clean_part(p) for p in parts if p and p != "."]
    return ("_" + "_".join(parts)) if parts else ""


def svg_to_png(src_svg: Path, dst_png: Path, size_px: int = 256) -> None:
    """
    Convert an SVG to a PNG to match other datasets.
    Tries (in order): cairosvg (python), rsvg-convert, inkscape.

    size_px is used as a target width/height. If a converter can't enforce both, it enforces width.
    """
    dst_png.parent.mkdir(parents=True, exist_ok=True)

    # 1) cairosvg (python)
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(
            url=str(src_svg),
            write_to=str(dst_png),
            output_width=size_px,
            output_height=size_px,
        )
        return
    except Exception:
        pass

    # 2) rsvg-convert (librsvg)
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", "-w", str(size_px), "-h", str(size_px), "-o", str(dst_png), str(src_svg)]
        subprocess.run(cmd, check=True)
        return

    # 3) inkscape
    if shutil.which("inkscape"):
        cmd = [
            "inkscape",
            str(src_svg),
            "--export-type=png",
            f"--export-filename={dst_png}",
            f"--export-width={size_px}",
            f"--export-height={size_px}",
        ]
        subprocess.run(cmd, check=True)
        return

    raise RuntimeError(
        "No SVG->PNG converter found. Install one of: python cairosvg, rsvg-convert (librsvg), or inkscape."
    )


# ------------------------- Dataset processors (MERGE-SAFE) -------------------------
def process_arasaac(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 1000
) -> Tuple[List[Dict[str, Any]], int]:
    print("\n[ARASAAC] Processing dataset...")

    metadata_file = ARASAAC_DIR / "metadata.json"
    if not metadata_file.exists():
        print(f"[WARN] ARASAAC metadata not found at {metadata_file}")
        return [], 0

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

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

        sub_sfx = subfolder_suffix(source_path, ARASAAC_DIR, strip_first="images")
        stem, suffix = Path(image_file).stem, Path(image_file).suffix
        new_filename = f"arasaac{sub_sfx}_{_clean_part(stem)}{suffix}"

        dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_path.name
        key = ("arasaac", final_filename)

        if key in existing_keys and dest_path.exists():
            continue

        if not dest_path.exists():
            try:
                move_preserve_meta(source_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {source_path} -> {dest_path}: {e}")
                continue

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


def process_aac(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 2000
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process AACIL scraped dataset (under AAC_DIR) and append only new entries (merge-safe).
    - Scans AAC_DIR recursively for image-like assets.
    - Prefixes output filenames with 'aac' + subfolder suffix to avoid collisions.
    """
    print("\n[AAC] Processing dataset...")

    if not AAC_DIR.exists():
        print(f"[WARN] AAC directory not found at {AAC_DIR}")
        return [], 0

    exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}
    processed_data: List[Dict[str, Any]] = []
    moved_count = 0
    seen = 0

    for src in AAC_DIR.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in exts:
            continue

        seen += 1

        sub_sfx = subfolder_suffix(src, AAC_DIR, strip_first=None)
        stem, suffix = src.stem, src.suffix
        new_filename = f"aac{sub_sfx}_{_clean_part(stem)}{suffix}"

        dest_path = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_path.name
        key = ("aac", final_filename)

        if key in existing_keys and dest_path.exists():
            if seen % save_interval == 0:
                print(f"[AAC] Checkpoint: scanned={seen}, moved={moved_count}")
            continue

        if not dest_path.exists():
            try:
                move_preserve_meta(src, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"[WARN] Failed to transfer {src} -> {dest_path}: {e}")
                continue

        if key not in existing_keys:
            try:
                rel = src.relative_to(AAC_DIR)
                rel_id = str(rel.with_suffix(""))
            except Exception:
                rel_id = src.stem

            title = src.stem.replace("_", " ").replace("-", " ").strip()
            keywords = [w for w in title.split() if w]

            entry = {
                "dataset": "aac",
                "image_file": final_filename,
                "id": rel_id,
                "title": title or src.stem,
                "keywords": keywords if keywords else [src.stem],
                "categories": ["aac"],
                "license": "See AACIL source",
                "skin_color": None,
                "hair_color": None,
                "background_color": None,
            }
            processed_data.append(entry)
            existing_keys.add(key)

        if seen % save_interval == 0:
            print(f"[AAC] Checkpoint: scanned={seen}, moved={moved_count}")

    print(f"[AAC] New entries: {len(processed_data)} | Files moved: {moved_count} | Assets scanned: {seen}")
    return processed_data, moved_count


def process_mulberry(
    all_metadata: List[Dict[str, Any]],
    existing_keys: set,
    save_interval: int = 2000,
    png_size_px: int = 256
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process Mulberry dataset:
    - Expects files directly in DATA_DIR/mulberry, like: "Afraid Man_3132.svg"
    - Converts SVG -> PNG to match other datasets.
    - Uses only the title from the filename (before the last underscore) as metadata.
    """
    print("\n[MULBERRY] Processing dataset...")

    if not MULBERRY_DIR.exists():
        print(f"[WARN] Mulberry directory not found at {MULBERRY_DIR}")
        return [], 0

    processed_data: List[Dict[str, Any]] = []
    converted_count = 0
    seen = 0

    for src_svg in MULBERRY_DIR.glob("*.svg"):
        if not src_svg.is_file():
            continue

        seen += 1

        # Examples:
        # "Afraid Man_3132.svg"
        # "Badger 2_3207.svg"
        base = src_svg.stem
        if "_" in base:
            title_part, id_part = base.rsplit("_", 1)
        else:
            title_part, id_part = base, base

        title = title_part.strip()
        sym_id = id_part.strip()

        # Convert to PNG and standardize naming
        safe_title = _clean_part(title.replace(" ", "_"))
        new_filename = f"mulberry_{safe_title}_{_clean_part(sym_id)}.png"

        dest_png = unique_dest(TRAINING_DIR / "images" / new_filename)
        final_filename = dest_png.name
        key = ("mulberry", final_filename)

        if key in existing_keys and dest_png.exists():
            if seen % save_interval == 0:
                print(f"[MULBERRY] Checkpoint: scanned={seen}, converted={converted_count}")
            continue

        if not dest_png.exists():
            try:
                svg_to_png(src_svg, dest_png, size_px=png_size_px)
                converted_count += 1
            except Exception as e:
                print(f"[WARN] Failed to convert {src_svg} -> {dest_png}: {e}")
                continue

        if key not in existing_keys:
            keywords = [w for w in title.replace("_", " ").split() if w]
            entry = {
                "dataset": "mulberry",
                "image_file": final_filename,
                "id": sym_id,
                "title": title,
                "keywords": keywords if keywords else [title],
                "categories": ["mulberry"],
                "license": "CC BY-SA 4.0",
                "skin_color": None,
                "hair_color": None,
                "background_color": None,
            }
            processed_data.append(entry)
            existing_keys.add(key)

        if seen % save_interval == 0:
            print(f"[MULBERRY] Checkpoint: scanned={seen}, converted={converted_count}")

    print(f"[MULBERRY] New entries: {len(processed_data)} | PNGs written: {converted_count} | SVGs scanned: {seen}")
    return processed_data, converted_count


# ---- your existing processors stay unchanged below this line ----
# process_icon645(...)
# process_lds(...)
# process_openmoji(...)
# save_metadata_with_backup(...)
# generate_statistics(...)


def main():
    print("=" * 60)
    print("PREPARING COMBINED DATASET (MERGE MODE)")
    print("=" * 60)

    setup_training_directory()

    all_metadata: List[Dict[str, Any]] = load_existing_metadata()
    existing_keys = build_existing_keys(all_metadata)
    grand_total = 0

    # AAC
    dataset_data, n = process_aac(all_metadata, existing_keys, save_interval=2000)
    all_metadata.extend(dataset_data)
    grand_total += n

    # Mulberry (SVG -> PNG)
    dataset_data, n = process_mulberry(all_metadata, existing_keys, save_interval=2000, png_size_px=256)
    all_metadata.extend(dataset_data)
    grand_total += n

    # ARASAAC
    dataset_data, n = process_arasaac(all_metadata, existing_keys, save_interval=1000)
    all_metadata.extend(dataset_data)
    grand_total += n

    # (the rest of your pipeline continues as before)
    dataset_data, n = process_icon645(all_metadata, existing_keys, save_interval=2000)
    all_metadata.extend(dataset_data)
    grand_total += n

    dataset_data, n = process_lds(all_metadata, existing_keys, save_interval=2000)
    all_metadata.extend(dataset_data)
    grand_total += n

    dataset_data, n = process_openmoji(all_metadata, existing_keys, save_interval=1000)
    all_metadata.extend(dataset_data)
    grand_total += n

    save_metadata_with_backup(all_metadata)
    generate_statistics(all_metadata)

    print("=" * 60)
    print(f"[SUCCESS] Merge complete!")
    print(f"[INFO] Training data location: {TRAINING_DIR}")
    print(f"[INFO] Files moved/converted this run: {grand_total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
