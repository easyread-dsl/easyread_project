#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate training prompts for each image in data/training_data using a stronger
local captioning model: Salesforce/blip2-flan-t5-xl (BLIP-2).

Features:
- Uses metadata hints (title, keywords, categories) to bias content.
- Skips ONLY entries already captioned by this exact model (prompt_model match).
- Saves and prints in BATCHES (not only at the end).
- Preserves existing metadata.

Notes:
- For speed/quality, run on GPU if available.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ------------------------- Config -------------------------

BATCH_SIZE = 50  # number of NEW prompts before saving/printing
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
PROMPT_MODEL_TAG = f"local:{MODEL_NAME}"

# Decoding config (tune for quality/speed)
MAX_NEW_TOKENS = 40
NUM_BEAMS = 4

# ------------------------- Paths -------------------------

HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / "../../data").resolve()

TRAINING_DIR = DATA_DIR / "training_data"
IMAGES_DIR = TRAINING_DIR / "images"
METADATA_JSON = TRAINING_DIR / "metadata.json"
METADATA_CSV = TRAINING_DIR / "metadata.csv"

# ------------------------- Model loading -------------------------

def load_blip2(device: torch.device):
    print(f"[INFO] Loading '{MODEL_NAME}' on device: {device}")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return processor, model

# ------------------------- Metadata helpers -------------------------

def load_metadata() -> List[Dict[str, Any]]:
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"metadata.json not found at {METADATA_JSON}")
    with METADATA_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def create_metadata_backups() -> None:
    if METADATA_JSON.exists():
        bak = METADATA_JSON.with_suffix(".json.bak")
        if not bak.exists():
            bak.write_bytes(METADATA_JSON.read_bytes())
            print(f"[INFO] Backup saved: {bak.name}")
    if METADATA_CSV.exists():
        bak = METADATA_CSV.with_suffix(".csv.bak")
        if not bak.exists():
            bak.write_bytes(METADATA_CSV.read_bytes())
            print(f"[INFO] Backup saved: {bak.name}")

def save_metadata_no_backup(all_metadata: List[Dict[str, Any]]) -> None:
    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    # CSV
    if not all_metadata:
        return
    fieldnames = set()
    for e in all_metadata:
        fieldnames.update(e.keys())
    preferred = ["dataset","image_file","id","title","keywords","categories","license","prompt","prompt_model"]
    ordered = preferred + [k for k in sorted(fieldnames) if k not in preferred]
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for e in all_metadata:
            row = e.copy()
            if isinstance(row.get("keywords"), list):
                row["keywords"] = "|".join(map(str, row["keywords"]))
            if isinstance(row.get("categories"), list):
                row["categories"] = "|".join(map(str, row["categories"]))
            w.writerow(row)
    print(f"[INFO] Saved JSON+CSV batch to disk.")

# ------------------------- Hint building -------------------------

def build_hint(meta: Dict[str, Any]) -> str:
    pieces = []
    t = meta.get("title")
    if t: pieces.append(str(t))
    kws = meta.get("keywords")
    if isinstance(kws, list) and kws:
        pieces.extend(map(str, kws))
    cats = meta.get("categories")
    if isinstance(cats, list) and cats:
        pieces.extend(map(str, cats))
    if not pieces:
        return ""
    # de-dupe while preserving order
    seen, uniq = set(), []
    for p in pieces:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return ", ".join(uniq)

# ------------------------- Prompt generation -------------------------

@torch.no_grad()
def generate_prompt_for_image(
    image_path: Path,
    meta: Dict[str, Any],
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    device: torch.device,
) -> Optional[str]:
    if not image_path.exists():
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    hint = build_hint(meta)
    # Instruction to keep it content-only (no style/quality fluff)
    if hint:
        q = (
            "Describe concisely what this image depicts. "
            "Only mention the semantic content (objects, people, actions). "
            "Do not mention style, quality, or rendering. "
            f"Use these hints only if they match the image: {hint}"
        )
    else:
        q = (
            "Describe concisely what this image depicts. "
            "Only mention the semantic content (objects, people, actions). "
            "Do not mention style, quality, or rendering."
        )

    inputs = processor(images=image, text=q, return_tensors="pt").to(device)

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            length_penalty=0.0,
            early_stopping=True,
        )
        text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if text:
            # light cleanup
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            return text
        return None
    except Exception:
        return None

# ------------------------- Main -------------------------

def main():
    print("=" * 60)
    print("GENERATING DIFFUSION PROMPTS (BLIP-2 FLAN-T5 XL • batched save/print)")
    print("=" * 60)

    if not TRAINING_DIR.exists():
        print(f"[FATAL] Training dir missing: {TRAINING_DIR}"); return
    if not IMAGES_DIR.exists():
        print(f"[FATAL] Images dir missing: {IMAGES_DIR}"); return
    if not METADATA_JSON.exists():
        print(f"[FATAL] metadata.json missing: {METADATA_JSON}"); return

    create_metadata_backups()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_blip2(device)

    all_meta = load_metadata()
    print(f"[INFO] Loaded {len(all_meta):,} entries.")

    total_updated = 0
    skipped_same_model = 0
    missing_images = 0

    batch_prompts: List[str] = []
    batch_updated = 0
    batch_index = 1

    for i, entry in enumerate(all_meta, start=1):
        # Skip ONLY if the prompt exists AND was produced by this same model
        existing_prompt = entry.get("prompt")
        existing_model = entry.get("prompt_model", "")
        if isinstance(existing_prompt, str) and existing_prompt.strip() and existing_model == PROMPT_MODEL_TAG:
            skipped_same_model += 1
            continue  # already captioned by this (better) model

        img_name = entry.get("image_file")
        if not img_name:
            continue

        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            missing_images += 1
            continue

        print(f"[INFO] [{i}/{len(all_meta)}] {img_name} -> generating prompt...")
        prompt = generate_prompt_for_image(img_path, entry, processor, model, device)
        if not prompt:
            # If generation fails, leave previous metadata untouched and move on
            print(f"[WARN] No prompt generated for {img_name}")
            continue

        entry["prompt"] = prompt
        entry["prompt_model"] = PROMPT_MODEL_TAG

        total_updated += 1
        batch_updated += 1
        batch_prompts.append(f"{img_name}: {prompt}")

        # Batch flush
        if batch_updated >= BATCH_SIZE:
            print("\n" + "-" * 60)
            print(f"[BATCH {batch_index}] New prompts (count = {batch_updated}):")
            for line in batch_prompts:
                print(f"[PROMPT] {line}")
            print("-" * 60)
            save_metadata_no_backup(all_meta)
            batch_prompts.clear()
            batch_updated = 0
            batch_index += 1

    # Flush remaining partial batch
    if batch_updated > 0:
        print("\n" + "-" * 60)
        print(f"[BATCH {batch_index}] New prompts (count = {batch_updated}):")
        for line in batch_prompts:
            print(f"[PROMPT] {line}")
        print("-" * 60)
        save_metadata_no_backup(all_meta)

    print("\n" + "-" * 60)
    print(f"[SUMMARY] Skipped (already BLIP-2 prompts): {skipped_same_model}")
    print(f"[SUMMARY] Missing images:                  {missing_images}")
    print(f"[SUMMARY] New prompts generated:           {total_updated}")
    print("-" * 60)
    print("[DONE] ✅ Prompts integrated into metadata.")
    print("=" * 60)

if __name__ == "__main__":
    main()
