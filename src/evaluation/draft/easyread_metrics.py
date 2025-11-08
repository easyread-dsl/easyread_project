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
import sys, os, math, argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math



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

# ---------- Utility helpers ----------
def draw_colored_contours(np_img, edges_binary):
    """
    Assign a distinct pseudo-random color to each external contour and render
    them over a black canvas (or over the original image if you prefer).
    edges_binary: uint8 binary image (0/255) containing the 'filled' mask used for contours.
    Returns an RGB uint8 image.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for contour visualization")

    contours, _ = cv2.findContours(edges_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = edges_binary.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Create a color per contour (HSV spaced hues → convert to RGB)
    n = max(1, len(contours))
    for i, c in enumerate(contours):
        hue = int((i / n) * 179)  # OpenCV HSV hue range [0..179]
        color_hsv = np.uint8([[[hue, 200, 255]]])  # vivid
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0,0,:].tolist()
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        cv2.drawContours(canvas, [c], -1, color_rgb, thickness=cv2.FILLED)

    return canvas

def palette_swatches_image(colors_rgb, chip=64, cols=None):
    """
    Build a swatch image for a given list/array of colors (Nx3, uint8).
    - chip: size of each color square in pixels
    - cols: how many chips per row (defaults to min(8, N))
    Returns a HxWx3 uint8 numpy image.
    """
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors_rgb must be of shape (N, 3) in uint8")

    N = colors.shape[0]
    if N == 0:
        # Return a small placeholder
        return np.full((chip, chip, 3), 255, dtype=np.uint8)

    if cols is None:
        cols = min(8, N)
    rows = (N + cols - 1) // cols

    H = rows * chip
    W = cols * chip
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    for i in range(N):
        r = i // cols
        c = i % cols
        y0, y1 = r * chip, (r + 1) * chip
        x0, x1 = c * chip, (c + 1) * chip
        canvas[y0:y1, x0:x1, :] = colors[i]

    return canvas

def save_visuals(image_path, np_img, palette_regions_colors=None, edges_img=None,
                 filled_for_contours=None, saliency_map=None,
                 saliency_mask=None, stroke_overlay=None, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    base = Path(image_path).stem

    # ---- Palette swatch from region colors only ----
    if palette_regions_colors is not None and len(palette_regions_colors) > 0:
        pdir = Path(out_dir) / "palettes"
        pdir.mkdir(parents=True, exist_ok=True)
        sw = palette_swatches_image(np.array(palette_regions_colors, dtype=np.uint8), chip=64, cols=None)
        Image.fromarray(sw).save(pdir / f"{base}_palette_regions.png")

    # ---- Edges & colored contours ----
    edir = Path(out_dir) / "edges"
    edir.mkdir(parents=True, exist_ok=True)
    if edges_img is not None:
        plt.imsave(edir / f"{base}_edges.png", edges_img, cmap="gray")
    if filled_for_contours is not None:
        try:
            contours_rgb = draw_colored_contours(np_img, filled_for_contours)
            Image.fromarray(contours_rgb).save(edir / f"{base}_contours_colored.png")
        except Exception as e:
            print(f"[warn] contour rendering failed: {e}")

    # ---- Saliency ----
    sdir = Path(out_dir) / "saliency"
    sdir.mkdir(parents=True, exist_ok=True)
    if saliency_map is not None:
        plt.imsave(sdir / f"{base}_saliency_map.png", saliency_map, cmap="inferno")
    if saliency_mask is not None:
        plt.imsave(sdir / f"{base}_saliency_mask.png", saliency_mask, cmap="gray")

    # ---- Strokes ----
    if stroke_overlay is not None:
        stdir = Path(out_dir) / "strokes"
        stdir.mkdir(parents=True, exist_ok=True)
        plt.imsave(stdir / f"{base}_strokes.png", stroke_overlay, cmap="magma")

    print(f"[Saved visuals to {out_dir}/ ]")



def load_image(path):
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required. Please install: pip install pillow")
    img = Image.open(path).convert("RGBA")
    # Composite over white background (change if your spec uses another bg)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img).convert("RGB")
    return comp

def pil_to_np(img_pil):
    return np.array(img_pil)  # HxWx3, uint8, sRGB

def to_gray(np_img):
    if cv2 is None:
        # fallback: manual luminance approximation
        r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
        return gray
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

def to_lab(np_img):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for LAB conversion: pip install opencv-python")
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)  # L in [0..255] approx

def robust_mean(arr, low_q=5, high_q=95):
    a = arr.reshape(-1).astype(np.float32)
    if a.size == 0:
        return float("nan")
    lo, hi = np.percentile(a, [low_q, high_q])
    sel = a[(a >= lo) & (a <= hi)]
    return float(sel.mean()) if sel.size else float(a.mean())

def safe_ratio(a, b):
    return float(a) / float(b) if float(b) != 0 else float('nan')

def mask_biggest_component(mask, min_area=0):
    # mask: uint8 binary {0,1}
    if cv2 is None:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = np.argmax(areas)
    if areas[idx] < min_area:
        return np.zeros_like(mask, dtype=np.uint8)
    biggest_label = idx + 1
    return (labels == biggest_label).astype(np.uint8)

# ---------- B) Palette size ----------

def quantize_palette_pil(img_pil, K):
    # Median-cut quantization via PIL
    q = img_pil.convert("P", palette=Image.ADAPTIVE, colors=K)
    # Get palette colors in RGB
    palette = q.getpalette()
    palette = np.array(palette, dtype=np.uint8).reshape(-1, 3)[:K]
    used = np.unique(np.array(q), return_counts=False)
    used_colors = palette[used]
    return q.convert("RGB"), used_colors

def palette_metrics(img_pil):
    """
    Returns:
      - palette_count_structural: smallest K with SSIM >= 0.98 (if SSIM available), else None
      - palette_count_regions: unique rounded colors with area threshold
      - small_area_color_fraction: % pixels in tiny color regions
      - used_palette_for_K: a dict mapping K to number of used colors
      - region_colors: (R,G,B) colors (uint8) that passed the area threshold (for a single swatch image)
      - region_counts: pixel counts for each region color (same order as region_colors)
    """
    np_img = pil_to_np(img_pil)
    H, W, _ = np_img.shape

    # Rounding to 5 bits per channel to merge near colors
    rounded = (np_img // 8) * 8  # 256/32 = 8

    # Count colors with pixel-count area filtering (0.1% image)
    colors, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    area_thresh = max(1, int(0.001 * H * W))
    big_mask = counts >= area_thresh
    k_regions = int(np.sum(big_mask))
    small_frac = float(np.sum(counts[~big_mask])) / float(H * W)

    # For visualization: return only the region-significant colors (sorted by frequency desc)
    region_colors = colors[big_mask]
    region_counts = counts[big_mask]
    if region_colors.shape[0] > 0:
        order = np.argsort(-region_counts)  # descending
        region_colors = region_colors[order]
        region_counts = region_counts[order]

    # Structural K via SSIM of quantized reconstructions (but do NOT save any palette images here)
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
        # If SSIM unavailable, still report how many palette entries are used at K=8
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
    img_resized = cv2.resize(np_img, (max(1, int(W*scale)), max(1, int(H*scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Otsu: retval is the scalar threshold, dst is the thresholded image
    t_otsu, _dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Build Canny thresholds around Otsu
    low  = max(0, int(round(0.66 * t_otsu)))
    high = min(255, int(round(1.33 * t_otsu)))
    if high <= low:
        # fallback guard if Otsu is degenerate
        low, high = 50, 150

    edges = cv2.Canny(gray, low, high)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    # Connect small gaps and suppress speckle
    edges_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    filled = cv2.morphologyEx(edges_dil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # External contours only; drop tiny ones
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = filled.shape[0] * filled.shape[1]
    min_area = max(1, int(0.0005 * area_img))  # 0.05%
    big_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Largest filled contour share
    largest_area = 0.0
    for c in big_contours:
        a = cv2.contourArea(c)
        if a > largest_area:
            largest_area = a
    largest_share = largest_area / area_img if area_img > 0 else float('nan')

    # Perimeter/area complexity
    perim_area_vals = []
    for c in big_contours:
        a = max(cv2.contourArea(c), 1.0)
        p = cv2.arcLength(c, True)
        perim_area_vals.append(p / a)
    perim_area_mean = float(np.mean(perim_area_vals)) if perim_area_vals else float('nan')

    return {
        "edge_density": float(edge_density),
        "num_contours": int(len(big_contours)),
        "largest_contour_share": float(largest_share),
        "perimeter_to_area_mean": perim_area_mean,
    }


# ---------- D) Saliency concentration ----------

def spectral_residual_saliency(gray):
    """
    Simple spectral residual saliency map (Hou & Zhang 2007 style).
    Returns normalized saliency map in [0,1].
    """
    g = gray.astype(np.float32) / 255.0
    if g.size == 0:
        return g
    h, w = g.shape
    # FFT
    G = np.fft.fft2(g)
    A = np.abs(G)
    L = np.log(A + 1e-8)
    # Average filter in frequency domain
    try:
        import scipy.ndimage as ndi
        L_avg = ndi.uniform_filter(L, size=3, mode='reflect')
    except Exception:
        # Fallback: naive 3x3 mean
        k = 3
        pad = k // 2
        Lp = np.pad(L, pad_width=pad, mode='edge')
        L_avg = np.zeros_like(L)
        for i in range(h):
            for j in range(w):
                L_avg[i, j] = Lp[i:i+k, j:j+k].mean()
    R = L - L_avg
    S = np.abs(np.fft.ifft2(np.exp(R + 1j * np.angle(G)))) ** 2
    if cv2 is not None:
        S = cv2.GaussianBlur(S.astype(np.float32), (9, 9), 2.5)
    # Normalize to [0,1]
    S -= S.min()
    if S.max() > 0:
        S /= S.max()
    return S

def get_saliency(np_img):
    gray = to_gray(np_img)
    try:
        S = spectral_residual_saliency(gray)
    except Exception:
        # Fallback: simple Laplacian magnitude as a poor proxy
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
    S = get_saliency(np_img)  # [0,1], float
    # Normalize S to sum=1
    Sm = S / (S.sum() + 1e-8)
    # Threshold to keep top X% mass
    flat = Sm.reshape(-1)
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cutoff_idx = np.searchsorted(cumsum, top_mass)
    thresh_val = flat[idx][cutoff_idx] if cutoff_idx < flat.size else flat[idx][-1]
    mask = (Sm >= thresh_val).astype(np.uint8)

    # Connected components on mask
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Compute mass per component
        masses = []
        for lbl in range(1, num_labels):
            comp_mass = float(Sm[labels == lbl].sum())
            masses.append(comp_mass)
        if len(masses) == 0:
            return {"saliency_concentration": float(0.0), "num_salient_blobs": 0,
                    "saliency_centroid": (float('nan'), float('nan')), "saliency_mask": mask}
        largest = max(masses)
        n_blobs = len(masses)
        # Centroid of the largest blob
        lbl_idx = 1 + np.argmax(masses)
        ys, xs = np.where(labels == lbl_idx)
        cx = float(np.mean(xs)) / mask.shape[1]
        cy = float(np.mean(ys)) / mask.shape[0]
        return {"saliency_concentration": float(largest / max(top_mass, 1e-8)),
                "num_salient_blobs": int(n_blobs),
                "saliency_centroid": (cx, cy),
                "saliency_mask": mask}
    else:
        # Fallback
        return {"saliency_concentration": float(top_mass), "num_salient_blobs": int(1),
                "saliency_centroid": (0.5, 0.5), "saliency_mask": mask}

# ---------- E) Foreground–background contrast ----------

def contrast_metrics(np_img, subject_mask=None):
    lab = to_lab(np_img)
    L = lab[..., 0].astype(np.float32)  # 0..255 approximately

    if subject_mask is None:
        # Use saliency (top 20% mass) as subject
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
        return {"delta_L": float('nan'), "fg_mean_L": float('nan'), "bg_mean_L": float('nan')}

    def rmean(x):
        if x.size == 0:
            return float('nan')
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
        return {"centroid": (float('nan'), float('nan')), "occupancy": 0.0, "centering_error": float('nan'),
                "bbox": (0, 0, 0, 0)}
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (xs.mean() + 0.5) / W
    cy = (ys.mean() + 0.5) / H
    occ = float(xs.size) / float(H * W)
    cent_err = max(abs(cx - 0.5), abs(cy - 0.5))
    bbox = (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
    return {"centroid": (float(cx), float(cy)), "occupancy": float(occ), "centering_error": float(cent_err),
            "bbox": bbox}

# ---------- I) Stroke / outline thickness ----------

def stroke_width_metrics(np_img):
    """
    Estimate stroke width from the foreground *interior* rather than grayscale edges:
    1) Adaptive binarization -> foreground (white).
    2) Euclidean distance transform on foreground.
    3) Sample radii at local maxima of the distance map (medial axis proxy).
    4) Fallback: also sample radii along the *foreground* boundary (Canny on bin_img).
    Returns median/IQR in px and normalized by image height.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for stroke metrics: pip install opencv-python")

    # 1) Robust binarization of the figure
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )  # foreground ~255

    # Heuristic: if very few foreground pixels, try the opposite inversion
    fg_ratio = np.count_nonzero(bin_img) / bin_img.size
    if fg_ratio < 0.01 or fg_ratio > 0.99:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
        )
        fg_ratio = np.count_nonzero(bin_img) / bin_img.size

    # 2) Distance transform on foreground
    dist = cv2.distanceTransform(bin_img, distanceType=cv2.DIST_L2, maskSize=5)  # float32 pixels

    # 3) Local-maxima mask (medial-axis proxy) -> radii from interior
    # Prefer scipy's maximum_filter; otherwise approximate with cv2.dilate on a uint8 proxy
    try:
        import scipy.ndimage as ndi
        max_f = ndi.maximum_filter(dist, size=3, mode="nearest")
        local_max_mask = (dist == max_f) & (dist > 0)
    except Exception:
        # Fallback: compare to 3x3 dilation of a scaled uint8 version
        if dist.max() > 0:
            dist8 = np.clip(dist / dist.max() * 255.0, 0, 255).astype(np.uint8)
        else:
            dist8 = dist.astype(np.uint8)
        dil = cv2.dilate(dist8, np.ones((3, 3), np.uint8), iterations=1)
        local_max_mask = (dist8 == dil) & (dist > 0)

    radii_core = dist[local_max_mask]

    # 4) Fallback: also sample along *foreground* boundary edges, not grayscale edges
    edges_fg = cv2.Canny(bin_img, 50, 150)  # edges around FG regions
    radii_edge = dist[edges_fg > 0]

    # Combine and clean
    radii = np.concatenate([radii_core, radii_edge]) if radii_core.size or radii_edge.size else np.array([], dtype=np.float32)
    if radii.size == 0:
        return {
            "stroke_median_px": float('nan'),
            "stroke_iqr_px": float('nan'),
            "relative_stroke_median": float('nan'),
            "relative_stroke_iqr": float('nan'),
        }

    # Remove zeros/negatives and extreme outliers
    radii = radii[np.isfinite(radii)]
    radii = radii[radii > 0]
    if radii.size == 0:
        return {
            "stroke_median_px": float('nan'),
            "stroke_iqr_px": float('nan'),
            "relative_stroke_median": float('nan'),
            "relative_stroke_iqr": float('nan'),
        }

    # Robust stats
    med_r = float(np.median(radii))
    q1, q3 = np.percentile(radii, [25, 75])
    iqr_r = float(q3 - q1)

    # Convert radius -> full stroke width (≈ 2 * radius)
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

def compute_metrics(image_path, save_visuals_flag=False):
    img_pil = load_image(image_path)
    np_img = pil_to_np(img_pil)

    # --- Palette metrics (and region colors for the single swatch image)
    palette = palette_metrics(img_pil)
    region_colors = palette.get("region_colors", [])

    # --- Edges: metrics + images needed for visualization
    try:
        edges_metrics = edge_simplicity(np_img)
    except Exception as e:
        edges_metrics = {"error": str(e)}

    edges_img, filled_for_contours = None, None
    if cv2 is not None:
        # Rebuild edge stack for visualization (same as edge_simplicity pipeline)
        H, W, _ = np_img.shape
        ref_width = 512
        scale = ref_width / float(W) if W else 1.0
        img_resized = cv2.resize(np_img, (max(1, int(W*scale)), max(1, int(H*scale))), interpolation=cv2.INTER_AREA)
        gray_vis = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        t_otsu, _ = cv2.threshold(gray_vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low, high = int(max(0, round(0.66 * t_otsu))), int(min(255, round(1.33 * t_otsu)))
        if high <= low:
            low, high = 50, 150
        edges_img = cv2.Canny(gray_vis, low, high)
        # Connect & close to get filled shapes (used to extract external contours)
        edges_dil = cv2.dilate(edges_img, np.ones((3, 3), np.uint8), iterations=1)
        filled_for_contours = cv2.morphologyEx(edges_dil, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    else:
        edges_img, filled_for_contours = None, None

    # --- Saliency
    sal_map = get_saliency(np_img)
    sal = saliency_concentration(np_img)
    sal_mask = sal.get("saliency_mask", None)

    # --- Contrast
    try:
        contrast = contrast_metrics(np_img, subject_mask=sal_mask)
    except Exception as e:
        contrast = {"error": str(e)}

    # --- Centering & Occupancy
    if sal_mask is not None:
        center_occ = centering_occupancy(sal_mask)
    else:
        center_occ = {"centroid": (float('nan'), float('nan')), "occupancy": float('nan'),
                      "centering_error": float('nan'), "bbox": (0, 0, 0, 0)}

    # --- Stroke width + visualization
    try:
        stroke = stroke_width_metrics(np_img)
    except Exception as e:
        stroke = {"error": str(e)}
    stroke_overlay = None
    if cv2 is not None:
        gray_s = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        bin_img = cv2.adaptiveThreshold(gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 35, 10)
        dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
        stroke_overlay = dist / (dist.max() + 1e-6)

    # Save visuals
    if save_visuals_flag:
        save_visuals(
            image_path,
            np_img,
            palette_regions_colors=np.array(region_colors, dtype=np.uint8),
            edges_img=edges_img,
            filled_for_contours=filled_for_contours,
            saliency_map=sal_map,
            saliency_mask=sal_mask,
            stroke_overlay=stroke_overlay,
            out_dir="results"
        )

    # Remove mask from metrics payload
    if "saliency_mask" in sal:
        sal.pop("saliency_mask", None)

    return {
        "palette": palette,
        "edges": edges_metrics,
        "saliency": sal,
        "contrast": contrast,
        "centering_occupancy": center_occ,
        "stroke": stroke,
    }



def exp_decay(val, a=5):
    """High when val is small, decays smoothly toward 0."""
    return math.exp(-a * max(val, 0))

def exp_rise(val, a=0.1):
    """Low when val is small, approaches 1 as val increases."""
    return 1 - math.exp(-a * max(val, 0))

def gauss_center(val, mu, sigma):
    """Peak at mu, falls off smoothly for deviations."""
    return math.exp(-((val - mu) ** 2) / (2 * sigma ** 2))

def normalize_metrics(raw):
    norm = {}

    # --- Palette ---
    # Simple linear normalization around (val - 3)/4, then exponential decay (lower = simpler = better)
    t = (raw["palette"]["palette_count_regions"] - 4.0) / 12.0
    norm["palette_count_regions"] = math.exp(-2.0 * max(t, 0.0))
    norm["small_area_color_fraction"] = math.exp(-3.0 * max(raw["palette"]["small_area_color_fraction"] / 0.02, 0.0))

    # --- Edge simplicity ---
    norm["edge_density"] = math.exp(-2.5 * max(raw["edges"]["edge_density"] / 0.1, 0.0))
    norm["num_contours"] = math.exp(-3.0 * max((raw["edges"]["num_contours"] - 1) / 4.0, 0.0))
    norm["largest_contour_share"] = math.exp(-((raw["edges"]["largest_contour_share"] - 0.4) ** 2) / (2 * 0.15 ** 2))
    # lower is better
    x = raw["edges"]["perimeter_to_area_mean"]
    t = x/0.15
    norm["perimeter_to_area_mean"] = math.exp(-2.0 * max(t, 0.0))

    # --- Saliency ---
    norm["saliency_concentration"] = 1 - math.exp(-1 * max(raw["saliency"]["saliency_concentration"], 0.0))
    norm["num_salient_blobs"] = math.exp(-2.0 * max((raw["saliency"]["num_salient_blobs"] - 1) / 2.0, 0.0))
    cx, cy = raw["saliency"]["saliency_centroid"]
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    norm["saliency_centroid"] = math.exp(-5.0 * max(dist / 0.5, 0.0))

    # --- Contrast ---
    norm["delta_L"] = 1 - math.exp(-3.0 * max(raw["contrast"]["delta_L"] / 120.0, 0.0))

    # --- Centering & occupancy ---
    occ = raw["centering_occupancy"]["occupancy"]
    norm["occupancy"] = math.exp(-((occ - 0.5) ** 2) / (2 * 0.2 ** 2))
    cenerr = raw["centering_occupancy"]["centering_error"]
    norm["centering_error"] = math.exp(-3.0 * max(cenerr / 0.5, 0.0))

    # --- Stroke ---
    rel = raw["stroke"]["relative_stroke_median"]
    norm["relative_stroke_median"] = math.exp(-((rel - 0.012) ** 2) / (2 * 0.004 ** 2))
    iqr = raw["stroke"]["stroke_iqr_px"]
    norm["stroke_iqr_px"] = math.exp(-2.0 * max(iqr / 2.0, 0.0))

    # Overall mean
    norm["overall_score"] = sum(norm.values()) / len(norm)
    return norm




def pretty_print(metrics):
    def fmt(x):
        if isinstance(x, float):
            if np.isnan(x):
                return "nan"
            return f"{x:.4f}"
        return str(x)

    print("\n=== EasyRead Style Metrics ===")
    # Palette
    p = metrics["palette"]
    print("\n[A] Palette size")
    print("  palette_count_structural:", p.get("palette_count_structural"))
    print("  palette_count_regions   :", p.get("palette_count_regions"))
    print("  small_area_color_fraction:", fmt(p.get("small_area_color_fraction")))
    if p.get("used_palette_for_K"):
        print("  used_palette_for_K      :", p.get("used_palette_for_K"))

    # Edges
    e = metrics["edges"]
    print("\n[B] Edge simplicity / line art")
    if "error" in e:
        print("  ERROR:", e["error"])
    else:
        print("  edge_density            :", fmt(e.get("edge_density", float('nan'))))
        print("  num_contours            :", e.get("num_contours"))
        print("  largest_contour_share   :", fmt(e.get("largest_contour_share", float('nan'))))
        print("  perimeter_to_area_mean  :", fmt(e.get("perimeter_to_area_mean", float('nan'))))

    # Saliency
    s = metrics["saliency"]
    print("\n[C] Saliency concentration")
    print("  saliency_concentration  :", fmt(s.get("saliency_concentration", float('nan'))))
    print("  num_salient_blobs       :", s.get("num_salient_blobs"))
    sc = s.get("saliency_centroid", (float('nan'), float('nan')))
    print("  saliency_centroid (x,y) :", (round(sc[0], 4), round(sc[1], 4)))

    # Contrast
    c = metrics["contrast"]
    print("\n[D] Foreground–background contrast")
    if "error" in c:
        print("  ERROR:", c["error"])
    else:
        print("  delta_L                 :", fmt(c.get("delta_L", float('nan'))))
        print("  fg_mean_L               :", fmt(c.get("fg_mean_L", float('nan'))))
        print("  bg_mean_L               :", fmt(c.get("bg_mean_L", float('nan'))))

    # Centering & Occupancy
    co = metrics["centering_occupancy"]
    print("\n[E] Centering & occupancy")
    cc = co.get("centroid", (float('nan'), float('nan')))
    print("  centroid (x,y)          :", (round(cc[0], 4), round(cc[1], 4)))
    print("  occupancy               :", fmt(co.get("occupancy", float('nan'))))
    print("  centering_error         :", fmt(co.get("centering_error", float('nan'))))
    print("  bbox (x,y,w,h)          :", co.get("bbox"))

    # Stroke
    st = metrics["stroke"]
    print("\n[F] Stroke / outline thickness")
    if "error" in st:
        print("  ERROR:", st["error"])
    else:
        print("  stroke_median_px        :", fmt(st.get("stroke_median_px", float('nan'))))
        print("  stroke_iqr_px           :", fmt(st.get("stroke_iqr_px", float('nan'))))
        print("  relative_stroke_median  :", fmt(st.get("relative_stroke_median", float('nan'))))
        print("  relative_stroke_iqr     :", fmt(st.get("relative_stroke_iqr", float('nan'))))

def main():
    parser = argparse.ArgumentParser(description="Compute EasyRead-style metrics for a single image (no AR/Resolution).")
    parser.add_argument("--image", default="../../../data/lds/accessible-changing-room-ventures.png", help="Path to input image")
    parser.add_argument("--json_out", type=str, default="results/metrics.json", help="Optional path to save metrics as JSON")
    parser.add_argument("--save_visuals", action="store_true",
                    help="If set, saves visualizations under results/")
    args = parser.parse_args()

    # 1. Compute raw metrics
    raw_metrics = compute_metrics(args.image, save_visuals_flag=args.save_visuals)

    # 2. Normalize with smooth functions
    norm_metrics = normalize_metrics(raw_metrics)

    # 3. Compute weighted EasyRead score
    palette_score = norm_metrics.get("palette_count_regions", 0.0)
    edge_score = norm_metrics.get("edge_density", 0.0)
    saliency_score = norm_metrics.get("saliency_concentration", 0.0)
    contrast_score = norm_metrics.get("delta_L", 0.0)
    stroke_score = norm_metrics.get("stroke_median_px", 0.0)
    centering_score = norm_metrics.get("centering_error", 0.0)

    easyread_score = (
        0.20 * palette_score
        + 0.15 * edge_score
        + 0.15 * saliency_score
        + 0.15 * contrast_score
        + 0.10 * stroke_score
        + 0.10 * centering_score
    )

    norm_metrics["EasyReadScore"] = easyread_score

    # 4. Print normalized metrics
    print("\n=== Normalized EasyRead Metrics ===")
    print(json.dumps(norm_metrics, indent=2))

    # 5. Nicely formatted breakdown
    print("\nEasyRead Score:")
    print("  Computed using weighted average of the following:")
    print("    1) palette_count_regions")
    print("    2) edge_density")
    print("    3) saliency_concentration")
    print("    4) delta_L")
    print("    5) stroke_median_px (relative stroke strength)")
    print("    6) centering_error")
    print("\n  EasyReadScore =")
    print("    0.20 * palette_score +")
    print("    0.15 * edge_score +")
    print("    0.15 * saliency_score +")
    print("    0.15 * contrast_score +")
    print("    0.10 * stroke_score +")
    print("    0.10 * centering_score\n")
    print(f"  Final EasyReadScore: {easyread_score:.3f}")

    # 6. Optionally save to JSON
    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(norm_metrics, f, indent=2)
        print(f"\nSaved normalized metrics to: {args.json_out}")


if __name__ == "__main__":
    main()