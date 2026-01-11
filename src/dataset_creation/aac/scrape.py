#!/usr/bin/env python3
"""
AACIL scraper/downloader (self-contained)

Goal:
- Start from https://aacil.neocities.org/
- Discover ALL internal HTML pages by crawling (BFS) from the homepage links
- Download every HTML page we find
- For each downloaded HTML page, download referenced assets:
  - <img src=...>
  - <link href=...> (CSS, favicon, etc.)
  - <script src=...> (only if hosted on the same domain; external scripts skipped)
- Mirror the website's path structure under: ../../data/aac/

This works on its own, without needing to specify --start.

Usage:
  python3 scrape_aac.py

Optional:
  python3 scrape_aac.py --max-pages 20000 --sleep 0.2
"""

from __future__ import annotations

import argparse
import os
import time
import json
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Deque, Iterable, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote, urldefrag

import requests


BASE_URL_DEFAULT = "https://aacil.neocities.org/"
DEFAULT_OUT_DIR = Path("../../../data/aac")  # as requested


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: Path
    status: str  # "downloaded" | "skipped" | "failed"
    http_status: Optional[int] = None
    error: Optional[str] = None


def is_same_site(url: str, base_url: str) -> bool:
    return urlparse(url).netloc == urlparse(base_url).netloc


def normalize_url(url: str) -> str:
    """Drop fragments and normalize a bit."""
    url, _frag = urldefrag(url)
    return url


def looks_like_html_path(path: str) -> bool:
    """
    Decide if a URL path likely points to an HTML page on AACIL.

    AACIL uses:
      /index (no ext) sometimes
      /map, /more, etc (no ext)
      many pages end with .html
    """
    if not path:
        return True
    if path.endswith("/"):
        return True
    ext = os.path.splitext(path)[1].lower()
    if ext in (".html", ".htm", ""):
        return True
    return False


def sanitize_local_path_from_url(url: str) -> Path:
    """
    Mirror the website path to a local file path.
    - "/" -> "index.html"
    - "/Body/body" -> "Body/body.html"
    - "/Body/body.html" -> "Body/body.html"
    - "/favicon.png" -> "favicon.png"
    """
    parsed = urlparse(url)
    path = unquote(parsed.path)

    if path in ("", "/"):
        path = "/index.html"

    root, ext = os.path.splitext(path)
    if not ext:
        if path.endswith("/"):
            path = path + "index.html"
        else:
            path = path + ".html"

    return Path(path.lstrip("/"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_text(dest: Path, text: str) -> None:
    ensure_parent(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, dest)


def download_binary(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    timeout: float,
    overwrite: bool,
) -> DownloadResult:
    try:
        if dest.exists() and not overwrite and dest.stat().st_size > 0:
            return DownloadResult(url=url, path=dest, status="skipped")

        ensure_parent(dest)
        with session.get(url, stream=True, timeout=timeout) as r:
            code = r.status_code
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
        return DownloadResult(url=url, path=dest, status="downloaded", http_status=code)
    except Exception as e:
        return DownloadResult(url=url, path=dest, status="failed", error=str(e))


def fetch_text(session: requests.Session, url: str, *, timeout: float) -> Optional[Tuple[int, str]]:
    try:
        r = session.get(url, timeout=timeout)
        code = r.status_code
        r.raise_for_status()
        # requests will guess encoding; if it fails, default to utf-8
        r.encoding = r.encoding or "utf-8"
        return code, r.text
    except Exception:
        return None


class LinkAssetParser(HTMLParser):
    """
    Extract:
    - internal links from <a href=...>
    - asset URLs from <img src=...>, <link href=...>, <script src=...>
    """
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: Set[str] = set()
        self.assets: Set[str] = set()

    def handle_starttag(self, tag: str, attrs):
        d = dict(attrs)

        if tag == "a":
            href = d.get("href")
            if href:
                self.hrefs.add(href.strip())

        if tag == "img":
            src = d.get("src")
            if src:
                self.assets.add(src.strip())

        if tag == "link":
            href = d.get("href")
            if href:
                self.assets.add(href.strip())

        if tag == "script":
            src = d.get("src")
            if src:
                self.assets.add(src.strip())


def parse_links_and_assets(html: str) -> LinkAssetParser:
    p = LinkAssetParser()
    p.feed(html)
    return p


def should_enqueue_link(abs_url: str, base_url: str) -> bool:
    if not is_same_site(abs_url, base_url):
        return False
    parsed = urlparse(abs_url)
    # Stay within site paths only
    if not parsed.path.startswith("/"):
        return False
    # Avoid obviously non-html assets
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js", ".json", ".pdf", ".zip"):
        return False
    # Accept html-like paths
    return looks_like_html_path(parsed.path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_URL_DEFAULT, help="AACIL base URL")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="Output dir (mirrors site structure)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Delay between requests (polite)")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    ap.add_argument("--max-pages", type=int, default=50000, help="Safety cap for number of HTML pages to crawl")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    base_url = args.base.rstrip("/") + "/"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "aacil-crawler/1.0 (polite; mirrors pages for offline research)"}
    )

    # BFS crawl queue
    q: Deque[str] = deque()
    seen_pages: Set[str] = set()
    seen_assets: Set[str] = set()

    start_url = base_url
    q.append(start_url)
    seen_pages.add(normalize_url(start_url))

    page_count = 0
    downloaded_pages = 0
    downloaded_assets = 0
    failed_pages = 0
    failed_assets = 0

    # Batch summary checkpointing (only affects the summary JSON, not file downloads)
    BATCH_EVERY_PAGES = 200

    def write_summary(path: Path) -> None:
        summary = {
            "base_url": base_url,
            "start_url": start_url,
            "out_dir": str(out_dir),
            "pages": {
                "seen": len(seen_pages),
                "processed": page_count,
                "downloaded": downloaded_pages,
                "failed": failed_pages,
                "queue_remaining": len(q),
            },
            "assets": {
                "seen": len(seen_assets),
                "downloaded": downloaded_assets,
                "failed": failed_assets,
            },
        }
        ensure_parent(path)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(
            f"[checkpoint] saved {path.name} | pages processed={page_count}, downloaded={downloaded_pages}, failed={failed_pages} | "
            f"assets downloaded={downloaded_assets}, failed={failed_assets}, seen={len(seen_assets)}"
        )

    while q and page_count < args.max_pages:
        page_url = q.popleft()
        page_url = normalize_url(page_url)

        fetched = fetch_text(session, page_url, timeout=args.timeout)
        page_count += 1

        if not fetched:
            failed_pages += 1
            if page_count % BATCH_EVERY_PAGES == 0:
                write_summary(out_dir / "download_summary.partial.json")
            time.sleep(args.sleep)
            continue

        http_code, html = fetched

        # Save HTML
        local_html = out_dir / sanitize_local_path_from_url(page_url)
        if args.overwrite or not local_html.exists():
            save_text(local_html, html)
        downloaded_pages += 1

        parsed = parse_links_and_assets(html)

        # Enqueue discovered internal pages
        for href in parsed.hrefs:
            abs_url = normalize_url(urljoin(page_url, href))
            if abs_url in seen_pages:
                continue
            if should_enqueue_link(abs_url, base_url):
                seen_pages.add(abs_url)
                q.append(abs_url)

        # Download assets (only same-site; skip external assets)
        for asset_ref in parsed.assets:
            abs_asset = normalize_url(urljoin(page_url, asset_ref))
            if abs_asset in seen_assets:
                continue
            seen_assets.add(abs_asset)

            if not is_same_site(abs_asset, base_url):
                # Skip external resources like the widget script on pages.dev
                continue

            dest = out_dir / sanitize_local_path_from_url(abs_asset)
            res = download_binary(
                session,
                abs_asset,
                dest,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            if res.status == "downloaded":
                downloaded_assets += 1
            elif res.status == "failed":
                failed_assets += 1

            time.sleep(args.sleep)

        if page_count % BATCH_EVERY_PAGES == 0:
            write_summary(out_dir / "download_summary.partial.json")

        time.sleep(args.sleep)

    summary = {
        "base_url": base_url,
        "start_url": start_url,
        "out_dir": str(out_dir),
        "pages": {
            "seen": len(seen_pages),
            "processed": page_count,
            "downloaded": downloaded_pages,
            "failed": failed_pages,
            "queue_remaining": len(q),
        },
        "assets": {
            "seen": len(seen_assets),
            "downloaded": downloaded_assets,
            "failed": failed_assets,
        },
    }

    # Write summary
    summary_path = out_dir / "download_summary.json"
    ensure_parent(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
