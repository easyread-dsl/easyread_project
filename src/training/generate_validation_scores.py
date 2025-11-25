#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze EasyRead + CLIP validation metrics for a LoRA run.

Usage:
    python analyze_validation_metrics.py /path/to/output_dir

Where /path/to/output_dir is the same as the training --output_dir.
This script will:

- Look for subdirectories named validation-<step> inside output_dir.
- For each image in each validation-<step> directory:
    - Compute EasyRead score via compute_easyread_score(image_path).
    - Compute CLIP text–image cosine similarity (prompt ~= filename).
- Save per-image metrics to:
    output_dir/easyread_validation_scores_recomputed.csv
- Save per-step average plots:
    output_dir/easyread_vs_step.png
    output_dir/clip_vs_step.png
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import csv

# -----------------------------------------------------------------------------
# Import compute_easyread_score from evaluation.easyread_metrics
# Assumes this file lives somewhere under src/, with:
#   src/evaluation/easyread_metrics.py
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent  # .../src

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from evaluation.easyread_metrics import compute_easyread_score
except Exception as e:
    raise ImportError(
        f"Failed to import compute_easyread_score from evaluation.easyread_metrics. "
        f"Make sure your repo has src/evaluation/easyread_metrics.py and that this "
        f"script is under src/ as well. Original error: {e}"
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

VAL_DIR_RE = re.compile(r"^validation-(\d+)$")


def find_validation_dirs(output_dir: Path) -> List[Tuple[int, Path]]:
    """Return list of (global_step, path_to_validation_dir), sorted by step."""
    dirs = []
    for child in output_dir.iterdir():
        if child.is_dir():
            m = VAL_DIR_RE.match(child.name)
            if m:
                step = int(m.group(1))
                dirs.append((step, child))
    dirs.sort(key=lambda x: x[0])
    return dirs


def infer_prompt_from_filename(fname: str) -> str:
    """
    Roughly reconstruct a prompt from the validation filename.

    E.g. "school_bus_at_stop.png" -> "school bus at stop"
    """
    stem = Path(fname).stem
    prompt = stem.replace("_", " ").strip()
    return prompt


def compute_clip_similarity(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    image: Image.Image,
    text: str,
) -> Optional[float]:
    """Compute cosine similarity between CLIP image/text embeddings."""
    try:
        inputs = clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds  # (1, d)
            text_embeds = outputs.text_embeds    # (1, d)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            sim = (image_embeds * text_embeds).sum(dim=-1).item()
        return float(sim)
    except Exception as e:
        print(f"[CLIP] Error computing similarity for text='{text}': {e}")
        return None




# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze_run(output_dir: Path) -> None:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir does not exist: {output_dir}")

    val_dirs = find_validation_dirs(output_dir)
    if not val_dirs:
        print(f"No validation-* directories found in {output_dir}")
        return

    print(f"Found {len(val_dirs)} validation directories.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP model '{clip_model_name}' on device: {device}")

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    # Per-image rows
    rows: List[Dict[str, object]] = []

    # Per-step aggregation
    per_step_scores: Dict[int, Dict[str, List[float]]] = {}

    for step, vdir in val_dirs:
        print(f"\n=== Processing validation step {step} at {vdir} ===")

        per_step_scores.setdefault(step, {"easyread": [], "clip": []})

        # Gather images (PNG/JPG/etc.)
        image_paths = [
            p for p in vdir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ]
        if not image_paths:
            print(f"[WARN] No images found in {vdir}")
            continue

        for img_path in image_paths:
            # Infer prompt from filename (approximate)
            prompt = infer_prompt_from_filename(img_path.name)

            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to open image {img_path}: {e}")
                continue

            # EasyRead score
            easy_score = None
            try:
                easy_score = float(compute_easyread_score(str(img_path)))
                per_step_scores[step]["easyread"].append(easy_score)
            except Exception as e:
                print(f"[EasyRead] Failed to compute score for {img_path}: {e}")

            # CLIP similarity
            clip_sim = compute_clip_similarity(
                clip_model,
                clip_processor,
                device,
                image,
                prompt,
            )


            if clip_sim is not None:
                per_step_scores[step]["clip"].append(clip_sim)

            rows.append(
                {
                    "global_step": step,
                    "prompt": prompt,
                    "image_path": str(img_path),
                    "easyread_score": easy_score,
                    "clip_similarity": clip_sim,
                }
            )

    # -------------------------------------------------------------------------
    # Save per-image CSV
    # -------------------------------------------------------------------------
    csv_path = output_dir / "easyread_validation_scores_recomputed.csv"
    print(f"\nSaving per-image metrics to {csv_path}")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["global_step", "prompt", "image_path", "easyread_score", "clip_similarity"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # -------------------------------------------------------------------------
    # Build per-step averages
    # -------------------------------------------------------------------------
    steps_sorted = sorted(per_step_scores.keys())
    avg_easyread = []
    avg_clip = []

    for step in steps_sorted:
        ez_list = per_step_scores[step]["easyread"]
        cl_list = per_step_scores[step]["clip"]

        if ez_list:
            avg_easy = float(np.mean(ez_list))
        else:
            avg_easy = float("nan")

        if cl_list:
            avg_cl = float(np.mean(cl_list))
        else:
            avg_cl = float("nan")

        avg_easyread.append(avg_easy)
        avg_clip.append(avg_cl)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    # EasyRead vs step
    plt.figure()
    plt.plot(steps_sorted, avg_easyread, marker="o")
    plt.xlabel("Global step")
    plt.ylabel("Average EasyRead score")
    plt.title("EasyRead score vs global step")
    plt.grid(True)
    easy_plot_path = output_dir / "easyread_vs_step.png"
    plt.tight_layout()
    plt.savefig(easy_plot_path)
    plt.close()
    print(f"Saved EasyRead evolution plot to {easy_plot_path}")

    # CLIP similarity vs step
    plt.figure()
    plt.plot(steps_sorted, avg_clip, marker="o")
    plt.xlabel("Global step")
    plt.ylabel("Average CLIP similarity")
    plt.title("CLIP text–image similarity vs global step")
    plt.grid(True)
    clip_plot_path = output_dir / "clip_vs_step.png"
    plt.tight_layout()
    plt.savefig(clip_plot_path)
    plt.close()
    print(f"Saved CLIP evolution plot to {clip_plot_path}")

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute EasyRead + CLIP metrics for all validation steps of a LoRA run."
    )
    ap.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory of a training run (same as --output_dir).",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    analyze_run(out_dir)
