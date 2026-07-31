#!/usr/bin/env python3
"""Download newly cataloged papers and rank architecture-figure captions.

This script is an editorial aid. It never decides which figure is published;
the ranked candidates must be verified visually against the source PDF.
"""

from __future__ import annotations

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


def read_manifest_entries() -> list[dict[str, object]]:
    text = README.read_text(encoding="utf-8")
    architecture_text = text.split("## Architectures", 1)[1]
    starts = list(re.finditer(r"^### \*\*(.+?)\*\*", architecture_text, re.M))
    sections = {}
    for position, match in enumerate(starts):
        end = starts[position + 1].start() if position + 1 < len(starts) else len(architecture_text)
        sections[match.group(1)] = architecture_text[match.start() : end]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    entries: list[dict[str, object]] = []
    for record in manifest:
        title = str(record["title"])
        if title not in sections:
            raise RuntimeError(f"missing architecture panel for manifest record: {title}")
        body = sections[title]
        ids = list(
            dict.fromkeys(
                re.findall(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", body)
            )
        )
        entries.append(
            {
                "index": record["index"],
                "title": title,
                "slug": record["slug"],
                "arxiv_ids": ids,
            }
        )
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
    score = sum(weight for word, weight in ARCHITECTURE_WORDS.items() if word in lowered)
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
    records = []
    failures = []
    for entry in read_manifest_entries():
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
            failures.append({"index": entry["index"], "arxiv_id": arxiv_id, "error": str(error)})
            records.append({**entry, "status": "failed", "error": str(error), "candidates": []})
            print(f"[{entry['index']:02}] FAILED {arxiv_id}: {error}", file=sys.stderr)
    OUTPUT.write_text(
        json.dumps({"records": records, "failures": failures}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(failures)} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
