#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

API_BASE = "https://api.arasaac.org/v1"
STATIC_BASE = "https://static.arasaac.org/pictograms"
LANG = "en"
OUT_DIR = (Path(__file__).resolve().parent / "../../../data/arasaac").resolve()

# Customization options for pictogram variations
SKIN_COLORS = ["white", "black", "assian", "mulatto", "aztec"]
HAIR_COLORS = ["blonde", "brown", "darkBrown", "gray", "darkGray", "red", "black"]
BACKGROUND_COLORS = {
    "red": "FF0000",
    "green": "00FF00",
    "blue": "0000FF",
    "yellow": "FFFF00",
    "black": "000000",
    "white": "FFFFFF",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; dataset-downloader/1.0)",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://arasaac.org/",
}

RETRY_STATUS = [403, 408, 409, 425, 429, 500, 502, 503, 504]
BACKOFF = 1.2
MAX_RETRIES = 5
SLEEP_BETWEEN_FILES = 0.25

# ---------- HTTP ----------
def make_session():
    s = requests.Session()
    r = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update(HEADERS)
    return s

def get_json(s: requests.Session, url: str, params=None):
    try:
        r = s.get(url, params=params, timeout=40)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def head_ok(s: requests.Session, url: str) -> bool:
    try:
        r = s.head(url, timeout=15, allow_redirects=True)
        if r.status_code == 200:
            return True
        # Some CDNs don’t support HEAD well; try a tiny ranged GET
        r = s.get(url, headers={"Range": "bytes=0-0"}, timeout=15)
        return r.status_code in (200, 206)
    except Exception:
        return False

# ---------- Catalog discovery (your working approach) ----------
def normalize_ids(raw) -> Optional[List[int]]:
    if not isinstance(raw, list) or not raw:
        return None
    # list of ints
    if all(isinstance(x, int) for x in raw):
        return sorted(set(int(x) for x in raw))
    # list of dicts with id / _id
    ids = []
    for item in raw:
        if isinstance(item, dict):
            if "id" in item and isinstance(item["id"], (int, str)):
                ids.append(int(item["id"]))
            elif "_id" in item and isinstance(item["_id"], (int, str)):
                ids.append(int(item["_id"]))
    return sorted(set(ids)) if ids else None

def try_endpoints(s: requests.Session, lang: str = LANG) -> Optional[List[int]]:
    candidates: List[str] = [
        f"{API_BASE}/pictograms/{lang}/all",
        f"{API_BASE}/pictograms/all?language={lang}",
        f"{API_BASE}/pictograms/all?lang={lang}",
        f"{API_BASE}/pictograms/all/{lang}",
        f"{API_BASE}/pictograms/all",
        # Some deployments expose a light list of ids:
        f"{API_BASE}/pictograms/ids",
        f"{API_BASE}/pictograms/ids?language={lang}",
        # Paginated variants:
        f"{API_BASE}/pictograms/{lang}",
        f"{API_BASE}/pictograms",
    ]
    for url in candidates:
        data = get_json(s, url)
        ids = normalize_ids(data)
        if ids:
            return ids
        # If paginated response encountered (list of dicts but small), try offset loop
        if isinstance(data, list) and url.endswith(f"/pictograms/{lang}"):
            ids = paginate_offset(s, base=f"{API_BASE}/pictograms/{lang}")
            if ids:
                return ids
    return None

def paginate_offset(s: requests.Session, base: str, limit: int = 500) -> Optional[List[int]]:
    off = 0
    all_items = []
    for _ in range(10000):
        data = get_json(s, base, params={"offset": off, "limit": limit})
        if not isinstance(data, list) or not data:
            break
        all_items.extend(data)
        off += limit
        time.sleep(0.15)
    return normalize_ids(all_items)

def probe_ids_by_head(s: requests.Session, lo: int, hi: int, step: int = 1) -> List[int]:
    """Bruteforce discover ids by probing the SVG URL. Careful: slow."""
    found = []
    for pid in range(lo, hi + 1, step):
        url = f"{STATIC_BASE}/{pid}/{pid}.svg"
        if head_ok(s, url):
            found.append(pid)
        if pid % 200 == 0:
            print(f"[PROBE] up to {pid}, found {len(found)}")
        time.sleep(0.03)
    return found

# ---------- File download ----------
# Global counter to rotate through variations
_variation_counter = 0

def get_next_variation_params(pid: int) -> Tuple[Dict[str, Any], str, str, str, str]:
    """
    Get the next variation parameters by rotating through each attribute independently.
    Each attribute cycles through its list at different rates for better diversity.
    Returns: (params_dict, skin_color, hair_color, bg_color_name, bg_color_hex)
    """
    global _variation_counter

    # Rotate each attribute independently - they cycle at different rates
    skin_idx = _variation_counter % len(SKIN_COLORS)
    hair_idx = _variation_counter % len(HAIR_COLORS)
    bg_idx = _variation_counter % len(BACKGROUND_COLORS)

    skin = SKIN_COLORS[skin_idx]
    hair = HAIR_COLORS[hair_idx]
    bg_name = list(BACKGROUND_COLORS.keys())[bg_idx]
    bg_hex = BACKGROUND_COLORS[bg_name]

    params = {
        "skin": skin,
        "hair": hair,
        "backgroundColor": bg_name,  # API expects color name, not hex
        "resolution": 500,
    }

    _variation_counter += 1
    return params, skin, hair, bg_name, bg_hex

def download_one(s: requests.Session, pid: int, out: Path) -> Tuple[bool, Optional[Path], Dict[str, Any]]:
    """
    Downloads one variation of a pictogram.
    Returns: (ok, saved_path, variation_metadata)
    """
    # Check if already downloaded
    filename = f"{pid}.png"
    dest = out / filename

    if dest.exists():
        # Try to infer variation from filename or just mark as existing
        variation_meta = {
            "file": filename,
            "skin": None,
            "hair": None,
            "background_color": None,
        }
        return True, dest, variation_meta

    # Get next variation parameters
    params, skin, hair, bg_name, bg_hex = get_next_variation_params(pid)

    # Build API URL
    url = f"{API_BASE}/pictograms/{pid}"

    try:
        r = s.get(url, params=params, timeout=45, stream=True)
        if r.status_code == 200:
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 14):
                    if chunk:
                        fh.write(chunk)

            variation_meta = {
                "file": filename,
                "skin": skin,
                "hair": hair,
                "background_color": bg_hex,
                "background_color_name": bg_name,
            }
            return True, dest, variation_meta
        return False, None, {}
    except Exception:
        return False, None, {}

# ---------- Metadata ----------
def _norm_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in seq:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out

def parse_keywords(obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    kw = obj.get("keywords")
    if isinstance(kw, list):
        for item in kw:
            if isinstance(item, dict) and "keyword" in item:
                k = _norm_text(item["keyword"])
                if k:
                    out.append(k)
            elif isinstance(item, str):
                k = _norm_text(item)
                if k:
                    out.append(k)
    tags = obj.get("tags")
    if isinstance(tags, list):
        for t in tags:
            k = _norm_text(t)
            if k:
                out.append(k)
    return _dedup(out)

def parse_categories(obj: Dict[str, Any]) -> List[str]:
    cats = obj.get("categories") or obj.get("category") or []
    if isinstance(cats, str):
        cats = [cats]
    out = []
    for c in cats:
        k = _norm_text(c)
        if k:
            out.append(k)
    return _dedup(out)

def fetch_metadata_for_id(s: requests.Session, pid: int, lang: str = LANG) -> Dict[str, Any]:
    """
    Try several endpoint shapes for per-id metadata; return {} on failure.
    """
    candidates = [
        f"{API_BASE}/pictograms/{lang}/{pid}",
        f"{API_BASE}/pictograms/{pid}?language={lang}",
        f"{API_BASE}/pictograms/{pid}?lang={lang}",
        f"{API_BASE}/pictograms/{pid}",
    ]
    data = None
    for url in candidates:
        data = get_json(s, url)
        if isinstance(data, dict) and data:
            break
        data = None

    if not data:
        return {}

    # Build normalized metadata row
    keywords = parse_keywords(data)
    title = keywords[0] if keywords else None
    meta = {
        "id": pid,
        "title": title,
        "keywords": keywords,
        "categories": parse_categories(data),
        "created": _norm_text(data.get("created") or data.get("dateCreation")),
        "updated": _norm_text(data.get("lastUpdated") or data.get("dateUpdate")),
        "schematic": data.get("schematic"),
        "hair": data.get("hair") or data.get("hairColor"),
        "skin": data.get("skin") or data.get("skinColor"),
        "api_lang": lang,
        # filled later: image_file, image_url
        "image_file": None,
        "image_url": None,
        "license": "CC BY-NC-SA 4.0 — © ARASAAC (Gobierno de Aragón), https://arasaac.org",
    }
    return meta

def save_metadata(meta_json_path: Path, meta_csv_path: Path, meta_map: Dict[int, Dict[str, Any]]) -> None:
    # JSON
    meta_json_path.write_text(json.dumps(meta_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV - One row per pictogram with variation details
    fields = [
        "id", "title", "keywords", "categories", "created", "updated",
        "schematic", "image_file", "skin_color", "hair_color", "background_color",
        "license", "api_lang"
    ]
    with meta_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for pid in sorted(meta_map.keys()):
            row = dict(meta_map[pid])
            # lists to pipe-separated
            if isinstance(row.get("keywords"), list):
                row["keywords"] = "|".join(row["keywords"])
            if isinstance(row.get("categories"), list):
                row["categories"] = "|".join(row["categories"])
            w.writerow(row)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Download ARASAAC pictograms + metadata")
    parser.add_argument("--lang", default=LANG, help="Language for keywords/titles (e.g., en, es, fr)")
    parser.add_argument("--probe-range", metavar=("LO", "HI"), type=int, nargs=2,
                        help="Use HEAD-probing fallback if the API list fails (inclusive range).")
    parser.add_argument("--limit", type=int, default=0, help="Max ids to process (0=all)")
    parser.add_argument("--resume", action="store_true", help="Reuse existing files and append/update metadata")
    parser.add_argument("--output-dir", type=str, help="Output directory for downloaded files (default: ../../../data/arasaac)")
    args = parser.parse_args()

    lang = args.lang.strip() or LANG
    out_dir = Path(args.output_dir).resolve() if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    s = make_session()

    print("[INFO] Fetching ARASAAC catalog …")
    ids = try_endpoints(s, lang=lang)

    if not ids and args.probe_range:
        lo, hi = args.probe_range
        print(f"[WARN] API catalog failed. Probing IDs {lo}..{hi} via CDN HEAD …")
        ids = probe_ids_by_head(s, lo, hi)

    if not ids:
        raise RuntimeError("Could not obtain ARASAAC catalog via API or probing.")

    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    print(f"[INFO] Catalog size: {len(ids):,} pictograms")
    (out_dir / "catalog_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_json_path = out_dir / "metadata.json"
    meta_csv_path = out_dir / "metadata.csv"
    meta_map: Dict[int, Dict[str, Any]] = {}

    # Load existing metadata if resuming
    if args.resume and meta_json_path.exists():
        try:
            meta_map = json.loads(meta_json_path.read_text(encoding="utf-8"))
            # keys as int
            meta_map = {int(k): v for k, v in meta_map.items()}
            print(f"[INFO] Loaded existing metadata for {len(meta_map)} items")
        except Exception:
            meta_map = {}

    ok = fail = 0

    for i, pid in enumerate(ids, 1):
        # fetch / update metadata first, so we always record something even if download fails
        if pid not in meta_map:
            meta_map[pid] = fetch_metadata_for_id(s, pid, lang=lang) or {"id": pid, "api_lang": lang}
            # ensure keys exist
            meta_map[pid].setdefault("title", None)
            meta_map[pid].setdefault("keywords", [])
            meta_map[pid].setdefault("categories", [])
            meta_map[pid].setdefault("created", None)
            meta_map[pid].setdefault("updated", None)
            meta_map[pid].setdefault("schematic", None)
            meta_map[pid].setdefault("license", "CC BY-NC-SA 4.0 — © ARASAAC (Gobierno de Aragón), https://arasaac.org")
            meta_map[pid].setdefault("image_file", None)
            meta_map[pid].setdefault("skin_color", None)
            meta_map[pid].setdefault("hair_color", None)
            meta_map[pid].setdefault("background_color", None)

        ok_dl, saved_path, variation_meta = download_one(s, pid, out_dir)

        if ok_dl:
            ok += 1
            # record file path and variation details
            if saved_path:
                meta_map[pid]["image_file"] = str(saved_path.relative_to(out_dir)).replace("\\", "/")
            meta_map[pid]["skin_color"] = variation_meta.get("skin")
            meta_map[pid]["hair_color"] = variation_meta.get("hair")
            meta_map[pid]["background_color"] = variation_meta.get("background_color_name") or variation_meta.get("background_color")

            if ok % 50 == 0:
                print(f"[OK] {ok:,} pictograms saved (last ID: {pid})")
        else:
            fail += 1
            print(f"[FAIL] ID {pid}")

        # periodic flush
        if i % 200 == 0 or i == len(ids):
            save_metadata(meta_json_path, meta_csv_path, meta_map)

        time.sleep(SLEEP_BETWEEN_FILES)

    # final save
    save_metadata(meta_json_path, meta_csv_path, meta_map)

    print(f"\n[DONE] Saved {ok:,} pictograms to {out_dir}")
    if fail:
        print(f"[WARN] {fail:,} pictograms could not be downloaded.")
    print(f"[DONE] Metadata -> {meta_json_path} and {meta_csv_path}")

if __name__ == "__main__":
    main()
