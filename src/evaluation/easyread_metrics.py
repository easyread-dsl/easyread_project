#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyRead Metrics Script (AR/Resolution removed)
Computes a set of style-and-layout metrics for simple/easyread icon-like images.

Metrics implemented:
- [B] Palette size (structural via SSIM when available; region-based unique colors with area threshold)
- [C] Edge simplicity (edge density, contour count, largest contour area share, perimeter/area complexity)
- [D] Saliency concentration (spectral residual saliency -> top-mass blobs)
- [E] Foreground–background contrast (ΔL* in LAB using subject mask)
- [F] Centering & occupancy (from subject mask)
- [I] Stroke/outline thickness (distance transform sampled at edges)

Requirements (install as needed):
  pip install pillow numpy opencv-python scipy scikit-image

Usage examples:
  python easyread_metrics.py --image path/to/img.png
  python easyread_metrics.py --image path/to/img.png --json_out out.json
"""
import os, math, argparse, json
import numpy as np


# ---------- Optional deps ----------
try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

# Optional: SSIM for palette estimation via structural preservation
try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False


# ---------- EasyRead score (fast path) ----------

def compute_easyread_score(image_path):
    """
    Compute the final scalar EasyReadScore for a single image.

    Uses:
      - palette_count_regions
      - edge_density
      - saliency_concentration
      - delta_L (foreground–background contrast)
      - relative_stroke_median (stroke thickness)
      - centering_error

    Only the metrics needed for this score are computed to save time.
    """
    img_pil = load_image(image_path)
    np_img = pil_to_np(img_pil)

    # Palette
    palette = palette_metrics(img_pil)
    k_regions = float(palette.get("palette_count_regions", 0.0))

    # Edges
    edges = edge_simplicity(np_img)
    edge_density = float(edges.get("edge_density", 0.0))

    # Saliency map, mask, concentration
    S = get_saliency(np_img)
    Sm = S / (S.sum() + 1e-8)

    flat = Sm.reshape(-1)
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    top_mass = 0.2
    cutoff_idx = np.searchsorted(cumsum, top_mass)
    thresh_val = flat[idx][cutoff_idx] if cutoff_idx < flat.size else flat[idx][-1]
    mask = (Sm >= thresh_val).astype(np.uint8)

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        masses = []
        for lbl in range(1, num_labels):
            comp_mass = float(Sm[labels == lbl].sum())
            masses.append(comp_mass)
        if len(masses) == 0:
            sal_conc = 0.0
        else:
            largest = max(masses)
            sal_conc = float(largest / max(top_mass, 1e-8))
    else:
        sal_conc = top_mass

    # Contrast using saliency-based subject mask
    contrast = contrast_metrics(np_img, subject_mask=mask)
    delta_L = float(contrast.get("delta_L", 0.0))

    # Centering & occupancy from same saliency mask
    center_occ = centering_occupancy(mask)
    centering_error = float(center_occ.get("centering_error", 0.0))

    # Stroke width
    stroke = stroke_width_metrics(np_img)
    rel_stroke = float(stroke.get("relative_stroke_median", 0.0))

    raw_metrics = {
        "palette": {"palette_count_regions": k_regions},
        "edges": {"edge_density": edge_density},
        "saliency": {"saliency_concentration": sal_conc},
        "contrast": {"delta_L": delta_L},
        "stroke": {"relative_stroke_median": rel_stroke},
        "centering_occupancy": {"centering_error": centering_error},
    }

    components = compute_easyread_components_from_raw(raw_metrics)
    return float(components["easyread_score"])


# ---------- Core helpers ----------

def load_image(path):
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required. Please install: pip install pillow")
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img).convert("RGB")
    return comp

def pil_to_np(img_pil):
    return np.array(img_pil)

def to_gray(np_img):
    if cv2 is None:
        r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
        return gray
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

def to_lab(np_img):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for LAB conversion: pip install opencv-python")
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)


# ---------- B) Palette size ----------

def quantize_palette_pil(img_pil, K):
    q = img_pil.convert("P", palette=Image.ADAPTIVE, colors=K)
    palette = q.getpalette()
    palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)[:K]
    used = np.unique(np.array(q), return_counts=False)
    used_colors = palette[used]
    return q.convert("RGB"), used_colors

def palette_metrics(img_pil):
    np_img = pil_to_np(img_pil)
    H, W, _ = np_img.shape

    rounded = (np_img // 8) * 8

    colors, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    area_thresh = max(1, int(0.001 * H * W))
    big_mask = counts >= area_thresh
    k_regions = int(np.sum(big_mask))
    small_frac = float(np.sum(counts[~big_mask])) / float(H * W)

    region_colors = colors[big_mask]
    region_counts = counts[big_mask]
    if region_colors.shape[0] > 0:
        order = np.argsort(-region_counts)
        region_colors = region_colors[order]
        region_counts = region_counts[order]

    Ks = [4, 6, 8, 10, 12, 16]
    used_palette_for_K = {}
    palette_count_structural = None
    if _HAS_SSIM and cv2 is not None:
        gray_orig = to_gray(np_img)
        for K in Ks:
            recon_pil, used_colors = quantize_palette_pil(img_pil, K)
            used_palette_for_K[int(K)] = int(used_colors.shape[0])
            recon_gray = to_gray(pil_to_np(recon_pil))
            try:
                s = ssim(gray_orig, recon_gray, data_range=255)
            except Exception:
                s = 0.0
            if s >= 0.98 and palette_count_structural is None:
                palette_count_structural = int(used_colors.shape[0])
    else:
        _, used_colors = quantize_palette_pil(img_pil, 8)
        used_palette_for_K[8] = int(used_colors.shape[0])

    return {
        "palette_count_structural": palette_count_structural,
        "palette_count_regions": int(k_regions),
        "small_area_color_fraction": float(small_frac),
        "used_palette_for_K": used_palette_for_K,
        "region_colors": region_colors.astype(np.uint8).tolist(),
        "region_counts": region_counts.astype(int).tolist(),
    }


# ---------- C) Edge simplicity / line art ----------

def edge_simplicity(np_img, ref_width=512):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for edge metrics: pip install opencv-python")
    H, W, _ = np_img.shape
    scale = ref_width / float(W) if W else 1.0
    img_resized = cv2.resize(np_img, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    t_otsu, _dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(0, int(round(0.66 * t_otsu)))
    high = min(255, int(round(1.33 * t_otsu)))
    if high <= low:
        low, high = 50, 150

    edges = cv2.Canny(gray, low, high)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    edges_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    filled = cv2.morphologyEx(edges_dil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = filled.shape[0] * filled.shape[1]
    min_area = max(1, int(0.0005 * area_img))
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    largest_area = 0.0
    for c in big_contours:
        a = cv2.contourArea(c)
        if a > largest_area:
            largest_area = a
    largest_share = largest_area / area_img if area_img > 0 else float("nan")

    perim_area_vals = []
    for c in big_contours:
        a = max(cv2.contourArea(c), 1.0)
        p = cv2.arcLength(c, True)
        perim_area_vals.append(p / a)
    perim_area_mean = float(np.mean(perim_area_vals)) if perim_area_vals else float("nan")

    return {
        "edge_density": float(edge_density),
        "num_contours": int(len(big_contours)),
        "largest_contour_share": float(largest_share),
        "perimeter_to_area_mean": perim_area_mean,
    }


# ---------- D) Saliency concentration ----------

def spectral_residual_saliency(gray):
    g = gray.astype(np.float32) / 255.0
    if g.size == 0:
        return g
    h, w = g.shape
    G = np.fft.fft2(g)
    A = np.abs(G)
    L = np.log(A + 1e-8)
    try:
        import scipy.ndimage as ndi
        L_avg = ndi.uniform_filter(L, size=3, mode="reflect")
    except Exception:
        k = 3
        pad = k // 2
        Lp = np.pad(L, pad_width=pad, mode="edge")
        L_avg = np.zeros_like(L)
        for i in range(h):
            for j in range(w):
                L_avg[i, j] = Lp[i:i + k, j:j + k].mean()
    R = L - L_avg
    S = np.abs(np.fft.ifft2(np.exp(R + 1j * np.angle(G)))) ** 2
    if cv2 is not None:
        S = cv2.GaussianBlur(S.astype(np.float32), (9, 9), 2.5)
    S -= S.min()
    if S.max() > 0:
        S /= S.max()
    return S

def get_saliency(np_img):
    gray = to_gray(np_img)
    try:
        S = spectral_residual_saliency(gray)
    except Exception:
        if cv2 is None:
            S = (gray.astype(np.float32) / 255.0)
        else:
            L = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            S = np.abs(L)
            S -= S.min()
            if S.max() > 0:
                S /= S.max()
    return S

def saliency_concentration(np_img, top_mass=0.2):
    S = get_saliency(np_img)
    Sm = S / (S.sum() + 1e-8)
    flat = Sm.reshape(-1)
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cutoff_idx = np.searchsorted(cumsum, top_mass)
    thresh_val = flat[idx][cutoff_idx] if cutoff_idx < flat.size else flat[idx][-1]
    mask = (Sm >= thresh_val).astype(np.uint8)

    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        masses = []
        for lbl in range(1, num_labels):
            comp_mass = float(Sm[labels == lbl].sum())
            masses.append(comp_mass)
        if len(masses) == 0:
            return {
                "saliency_concentration": float(0.0),
                "num_salient_blobs": 0,
                "saliency_centroid": (float("nan"), float("nan")),
                "saliency_mask": mask,
            }
        largest = max(masses)
        n_blobs = len(masses)
        lbl_idx = 1 + np.argmax(masses)
        ys, xs = np.where(labels == lbl_idx)
        cx = float(np.mean(xs)) / mask.shape[1]
        cy = float(np.mean(ys)) / mask.shape[0]
        return {
            "saliency_concentration": float(largest / max(top_mass, 1e-8)),
            "num_salient_blobs": int(n_blobs),
            "saliency_centroid": (cx, cy),
            "saliency_mask": mask,
        }
    else:
        return {
            "saliency_concentration": float(top_mass),
            "num_salient_blobs": int(1),
            "saliency_centroid": (0.5, 0.5),
            "saliency_mask": mask,
        }


# ---------- E) Foreground–background contrast ----------

def contrast_metrics(np_img, subject_mask=None):
    lab = to_lab(np_img)
    L = lab[..., 0].astype(np.float32)

    if subject_mask is None:
        sal = get_saliency(np_img)
        flat = sal.reshape(-1)
        idx = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[idx] / (flat.sum() + 1e-8))
        cutoff_idx = np.searchsorted(cumsum, 0.2)
        thresh_val = flat[idx][cutoff_idx] if cutoff_idx < flat.size else flat[idx][-1]
        subject_mask = (sal >= thresh_val).astype(np.uint8)

    subject_mask = subject_mask.astype(np.uint8)
    bg_mask = (1 - subject_mask).astype(np.uint8)

    L_fg = L[subject_mask == 1]
    L_bg = L[bg_mask == 1]
    if L_fg.size == 0 or L_bg.size == 0:
        return {"delta_L": float("nan"), "fg_mean_L": float("nan"), "bg_mean_L": float("nan")}

    def rmean(x):
        if x.size == 0:
            return float("nan")
        lo, hi = np.percentile(x, [5, 95])
        sel = x[(x >= lo) & (x <= hi)]
        return float(sel.mean()) if sel.size else float(x.mean())

    fg_mean = rmean(L_fg)
    bg_mean = rmean(L_bg)
    delta_L = abs(fg_mean - bg_mean)

    return {"delta_L": float(delta_L), "fg_mean_L": float(fg_mean), "bg_mean_L": float(bg_mean)}


# ---------- F) Centering & occupancy ----------

def centering_occupancy(subject_mask):
    H, W = subject_mask.shape
    ys, xs = np.where(subject_mask > 0)
    if xs.size == 0:
        return {
            "centroid": (float("nan"), float("nan")),
            "occupancy": 0.0,
            "centering_error": float("nan"),
            "bbox": (0, 0, 0, 0),
        }
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (xs.mean() + 0.5) / W
    cy = (ys.mean() + 0.5) / H
    occ = float(xs.size) / float(H * W)
    cent_err = max(abs(cx - 0.5), abs(cy - 0.5))
    bbox = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
    return {
        "centroid": (float(cx), float(cy)),
        "occupancy": float(occ),
        "centering_error": float(cent_err),
        "bbox": bbox,
    }


# ---------- I) Stroke / outline thickness ----------

def stroke_width_metrics(np_img):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for stroke metrics: pip install opencv-python")

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )

    fg_ratio = np.count_nonzero(bin_img) / bin_img.size
    if fg_ratio < 0.01 or fg_ratio > 0.99:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )

    dist = cv2.distanceTransform(bin_img, distanceType=cv2.DIST_L2, maskSize=5)

    try:
        import scipy.ndimage as ndi
        max_f = ndi.maximum_filter(dist, size=3, mode="nearest")
        local_max_mask = (dist == max_f) & (dist > 0)
    except Exception:
        if dist.max() > 0:
            dist8 = np.clip(dist / dist.max() * 255.0, 0, 255).astype(np.uint8)
        else:
            dist8 = dist.astype(np.uint8)
        dil = cv2.dilate(dist8, np.ones((3, 3), np.uint8), iterations=1)
        local_max_mask = (dist8 == dil) & (dist > 0)

    radii_core = dist[local_max_mask]

    edges_fg = cv2.Canny(bin_img, 50, 150)
    radii_edge = dist[edges_fg > 0]

    radii = np.concatenate([radii_core, radii_edge]) if (radii_core.size or radii_edge.size) else np.array([], dtype=np.float32)
    if radii.size == 0:
        return {
            "stroke_median_px": float("nan"),
            "stroke_iqr_px": float("nan"),
            "relative_stroke_median": float("nan"),
            "relative_stroke_iqr": float("nan"),
        }

    radii = radii[np.isfinite(radii)]
    radii = radii[radii > 0]
    if radii.size == 0:
        return {
            "stroke_median_px": float("nan"),
            "stroke_iqr_px": float("nan"),
            "relative_stroke_median": float("nan"),
            "relative_stroke_iqr": float("nan"),
        }

    med_r = float(np.median(radii))
    q1, q3 = np.percentile(radii, [25, 75])
    iqr_r = float(q3 - q1)

    stroke_median = 2.0 * med_r
    stroke_iqr = 2.0 * iqr_r

    H = np_img.shape[0]
    rel_med = stroke_median / float(H)
    rel_iqr = stroke_iqr / float(H)

    return {
        "stroke_median_px": float(stroke_median),
        "stroke_iqr_px": float(stroke_iqr),
        "relative_stroke_median": float(rel_med),
        "relative_stroke_iqr": float(rel_iqr),
    }


# ---------- Orchestration ----------

def compute_metrics(image_path):
    img_pil = load_image(image_path)
    np_img = pil_to_np(img_pil)

    palette = palette_metrics(img_pil)
    edges = edge_simplicity(np_img)

    sal = saliency_concentration(np_img)
    sal_mask = sal.get("saliency_mask", None)

    contrast = contrast_metrics(np_img, subject_mask=sal_mask) if sal_mask is not None else contrast_metrics(np_img)

    if sal_mask is not None:
        center_occ = centering_occupancy(sal_mask)
    else:
        center_occ = {
            "centroid": (float("nan"), float("nan")),
            "occupancy": float("nan"),
            "centering_error": float("nan"),
            "bbox": (0, 0, 0, 0),
        }

    stroke = stroke_width_metrics(np_img)

    if "saliency_mask" in sal:
        sal.pop("saliency_mask", None)

    return {
        "palette": palette,
        "edges": edges,
        "saliency": sal,
        "contrast": contrast,
        "centering_occupancy": center_occ,
        "stroke": stroke,
    }


# ---------- Normalization & scoring ----------

def normalize_metrics(raw):
    norm = {}

    # Palette
    t = (raw["palette"]["palette_count_regions"] - 4.0) / 12.0
    norm["palette_count_regions"] = math.exp(-2.0 * max(t, 0.0))
    norm["small_area_color_fraction"] = math.exp(
        -3.0 * max(raw["palette"]["small_area_color_fraction"] / 0.02, 0.0)
    )

    # Edges
    norm["edge_density"] = math.exp(-2.5 * max(raw["edges"]["edge_density"] / 0.1, 0.0))
    norm["num_contours"] = math.exp(-3.0 * max((raw["edges"]["num_contours"] - 1) / 4.0, 0.0))
    norm["largest_contour_share"] = math.exp(
        -((raw["edges"]["largest_contour_share"] - 0.4) ** 2) / (2 * 0.15 ** 2)
    )
    x = raw["edges"]["perimeter_to_area_mean"]
    t = x / 0.15
    norm["perimeter_to_area_mean"] = math.exp(-2.0 * max(t, 0.0))

    # Saliency
    norm["saliency_concentration"] = 1 - math.exp(
        -1 * max(raw["saliency"]["saliency_concentration"], 0.0)
    )
    norm["num_salient_blobs"] = math.exp(
        -2.0 * max((raw["saliency"]["num_salient_blobs"] - 1) / 2.0, 0.0)
    )
    cx, cy = raw["saliency"]["saliency_centroid"]
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    norm["saliency_centroid"] = math.exp(-5.0 * max(dist / 0.5, 0.0))

    # Contrast
    norm["delta_L"] = 1 - math.exp(-3.0 * max(raw["contrast"]["delta_L"] / 120.0, 0.0))

    # Centering & occupancy
    occ = raw["centering_occupancy"]["occupancy"]
    norm["occupancy"] = math.exp(-((occ - 0.5) ** 2) / (2 * 0.2 ** 2))
    cenerr = raw["centering_occupancy"]["centering_error"]
    norm["centering_error"] = math.exp(-3.0 * max(cenerr / 0.5, 0.0))

    # Stroke
    rel = raw["stroke"]["relative_stroke_median"]
    norm["relative_stroke_median"] = math.exp(
        -((rel - 0.012) ** 2) / (2 * 0.004 ** 2)
    )
    iqr = raw["stroke"]["stroke_iqr_px"]
    norm["stroke_iqr_px"] = math.exp(-2.0 * max(iqr / 2.0, 0.0))

    norm["overall_score"] = sum(norm.values()) / len(norm)
    return norm


def compute_palette_score_from_raw(raw_metrics):
    k_regions = float(raw_metrics["palette"]["palette_count_regions"])
    t_pal = (k_regions - 4.0) / 12.0
    return math.exp(-2.0 * max(t_pal, 0.0))

def compute_edge_score_from_raw(raw_metrics):
    edge_density = float(raw_metrics["edges"]["edge_density"])
    return math.exp(-2.5 * max(edge_density / 0.1, 0.0))

def compute_saliency_score_from_raw(raw_metrics):
    sal_conc = float(raw_metrics["saliency"]["saliency_concentration"])
    return 1.0 - math.exp(-1.0 * max(sal_conc, 0.0))

def compute_contrast_score_from_raw(raw_metrics):
    delta_L = float(raw_metrics["contrast"]["delta_L"])
    return 1.0 - math.exp(-3.0 * max(delta_L / 120.0, 0.0))

def compute_stroke_score_from_raw(raw_metrics):
    rel_stroke = float(raw_metrics["stroke"]["relative_stroke_median"])
    return math.exp(-((rel_stroke - 0.012) ** 2) / (2.0 * 0.004 ** 2))

def compute_centering_score_from_raw(raw_metrics):
    centering_error = float(raw_metrics["centering_occupancy"]["centering_error"])
    return math.exp(-3.0 * max(centering_error / 0.5, 0.0))

def compute_easyread_components_from_raw(raw_metrics):
    palette_score = compute_palette_score_from_raw(raw_metrics)
    edge_score = compute_edge_score_from_raw(raw_metrics)
    saliency_score = compute_saliency_score_from_raw(raw_metrics)
    contrast_score = compute_contrast_score_from_raw(raw_metrics)
    stroke_score = compute_stroke_score_from_raw(raw_metrics)
    centering_score = compute_centering_score_from_raw(raw_metrics)

    easyread_score = (
        0.25 * palette_score
        + 0.20 * edge_score
        + 0.15 * saliency_score
        + 0.15 * contrast_score
        + 0.15 * stroke_score
        + 0.10 * centering_score
    )

    return {
        "palette_score": float(palette_score),
        "edge_score": float(edge_score),
        "saliency_score": float(saliency_score),
        "contrast_score": float(contrast_score),
        "stroke_score": float(stroke_score),
        "centering_score": float(centering_score),
        "easyread_score": float(easyread_score),
    }


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Compute EasyRead-style metrics for a single image (no AR/Resolution)."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="results/metrics.json",
        help="Optional path to save metrics as JSON",
    )
    args = parser.parse_args()

    raw_metrics = compute_metrics(args.image)
    norm_metrics = normalize_metrics(raw_metrics)
    components = compute_easyread_components_from_raw(raw_metrics)

    norm_metrics["EasyReadScore"] = components["easyread_score"]

    print("\n=== Normalized EasyRead Metrics ===")
    print(json.dumps(norm_metrics, indent=2))

    print("\nEasyRead Score (from raw metrics):")
    print("  Components:")
    for k in [
        "palette_score",
        "edge_score",
        "saliency_score",
        "contrast_score",
        "stroke_score",
        "centering_score",
    ]:
        print(f"    {k}: {components[k]:.4f}")
    print(f"\n  Final EasyReadScore: {components['easyread_score']:.3f}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_metrics": raw_metrics,
                    "normalized_metrics": norm_metrics,
                    "easyread_components": components,
                },
                f,
                indent=2,
            )
        print(f"\nSaved metrics to: {args.json_out}")


if __name__ == "__main__":
    main()
