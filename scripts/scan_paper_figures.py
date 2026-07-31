#!/usr/bin/env python3
"""Download newly cataloged papers and rank architecture-figure captions.

This script is an editorial aid. It never decides which figure is published;
the ranked candidates must be verified visually against the source PDF.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PDF_DIR = ROOT / "tmp" / "pdfs"
OUTPUT = PDF_DIR / "candidates.json"
MANIFEST = ROOT / "assets" / "architectures" / "manifest.json"

ARCHITECTURE_WORDS = {
    "architecture": 12,
    "architectural": 12,
    "overview": 10,
    "framework": 9,
    "model": 4,
    "pipeline": 7,
    "network": 5,
    "training": 3,
    "attention": 4,
    "encoder": 4,
    "decoder": 4,
    "transformer": 4,
    "system": 3,
    "method": 3,
}
NEGATIVE_WORDS = {
    "benchmark": -8,
    "results": -7,
    "accuracy": -6,
    "comparison": -5,
    "examples": -5,
    "visualization": -3,
}


def slugify(title: str) -> str:
    """Create a readable, deterministic asset slug from an architecture title."""
    short_title = re.split(r":|\s+-\s+", title, maxsplit=1)[0]
    value = short_title.lower().replace("+", " plus ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:72].rstrip("-")


def read_panel_entries(scope: str) -> list[dict[str, object]]:
    text = README.read_text(encoding="utf-8")
    architecture_text = text.split("## Architectures", 1)[1].split(
        "## Important References", 1
    )[0]
    starts = list(re.finditer(r"^### \*\*(.+?)\*\*", architecture_text, re.M))
    panels: list[tuple[str, str]] = []
    for position, match in enumerate(starts):
        end = (
            starts[position + 1].start()
            if position + 1 < len(starts)
            else len(architecture_text)
        )
        panels.append((match.group(1), architecture_text[match.start() : end]))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    by_title = {str(record["title"]): record for record in manifest}
    if len(by_title) != len(manifest):
        raise RuntimeError("duplicate titles in architecture figure manifest")
    panel_titles = {title for title, _ in panels}
    missing = sorted(set(by_title) - panel_titles)
    if missing:
        raise RuntimeError(
            f"missing architecture panels for manifest records: {missing}"
        )

    next_index = max((int(record["index"]) for record in manifest), default=0) + 1
    entries: list[dict[str, object]] = []
    for catalog_position, (title, body) in enumerate(panels, start=1):
        record = by_title.get(title)
        is_manifest = record is not None
        if scope == "manifest" and not is_manifest:
            continue
        if scope == "legacy" and is_manifest:
            continue
        ids = list(
            dict.fromkeys(re.findall(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", body))
        )
        links = list(dict.fromkeys(re.findall(r"https?://[^)\s\"<>]+", body)))
        current_images = re.findall(
            r"(?:<img\s+[^>]*src=\"|!\[[^]]*\]\()([^\")]+)", body
        )
        entries.append(
            {
                "index": int(record["index"]) if record else next_index,
                "catalog_position": catalog_position,
                "title": title,
                "slug": str(record["slug"]) if record else slugify(title),
                "manifest_status": str(record["status"]) if record else None,
                "arxiv_ids": ids,
                "panel_links": links,
                "current_images": current_images,
            }
        )
        if not record:
            next_index += 1
    return entries


def download_pdf(arxiv_id: str) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    target = PDF_DIR / f"{arxiv_id}.pdf"
    if target.exists() and target.stat().st_size > 50_000:
        return target
    temporary = target.with_suffix(".pdf.part")
    request = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": "awesome-vlm-architectures/figure-audit"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        temporary.write_bytes(response.read())
    temporary.replace(target)
    time.sleep(0.15)
    return target


def normalize_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def caption_score(caption: str, figure_number: str) -> int:
    lowered = caption.lower()
    score = sum(
        weight for word, weight in ARCHITECTURE_WORDS.items() if word in lowered
    )
    score += sum(weight for word, weight in NEGATIVE_WORDS.items() if word in lowered)
    try:
        score += max(0, 6 - int(re.match(r"\d+", figure_number).group()))
    except (AttributeError, ValueError):
        pass
    return score


def scan_pdf(path: Path) -> list[dict[str, object]]:
    document = fitz.open(path)
    candidates: list[dict[str, object]] = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Figure|Fig\.)\s*([0-9]+(?:[a-z])?)\s*[:.\-–—]?\s*(.+)",
        re.I,
    )
    for page_index, page in enumerate(document):
        for block in page.get_text("blocks", sort=True):
            block_text = normalize_caption(block[4])
            for match in pattern.finditer(block[4]):
                caption = normalize_caption(match.group(0))
                if len(caption) < 18:
                    caption = block_text
                number = match.group(1)
                candidates.append(
                    {
                        "figure": number,
                        "pdf_page": page_index + 1,
                        "caption": caption[:900],
                        "score": caption_score(caption, number),
                        "caption_bbox": [round(value, 2) for value in block[:4]],
                    }
                )
    candidates.sort(key=lambda item: (-int(item["score"]), int(item["pdf_page"])))
    return candidates[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("legacy", "manifest", "all"),
        default="legacy",
        help="panels to scan (default: legacy panels not yet in the manifest)",
    )
    parser.add_argument("--start", type=int, help="first catalog position to include")
    parser.add_argument("--end", type=int, help="last catalog position to include")
    parser.add_argument("--limit", type=int, help="maximum number of panels to scan")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="candidate JSON destination (default: tmp/pdfs/candidates.json)",
    )
    args = parser.parse_args()

    entries = read_panel_entries(args.scope)
    if args.start is not None:
        entries = [
            entry for entry in entries if int(entry["catalog_position"]) >= args.start
        ]
    if args.end is not None:
        entries = [
            entry for entry in entries if int(entry["catalog_position"]) <= args.end
        ]
    if args.limit is not None:
        entries = entries[: args.limit]

    records = []
    failures = []
    for entry in entries:
        ids = entry["arxiv_ids"]
        if not ids:
            records.append({**entry, "status": "no-arxiv", "candidates": []})
            continue
        arxiv_id = str(ids[0])
        try:
            path = download_pdf(arxiv_id)
            records.append(
                {
                    **entry,
                    "status": "scanned",
                    "selected_arxiv_id": arxiv_id,
                    "pdf": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "candidates": scan_pdf(path),
                }
            )
            print(f"[{entry['index']:02}] scanned {arxiv_id} - {entry['title']}")
        except Exception as error:  # noqa: BLE001 - continue the batch and report all failures
            failures.append(
                {"index": entry["index"], "arxiv_id": arxiv_id, "error": str(error)}
            )
            records.append(
                {**entry, "status": "failed", "error": str(error), "candidates": []}
            )
            print(f"[{entry['index']:02}] FAILED {arxiv_id}: {error}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"records": records, "failures": failures}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        output_label = args.output.relative_to(ROOT)
    except ValueError:
        output_label = args.output
    print(f"Wrote {output_label} with {len(failures)} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
