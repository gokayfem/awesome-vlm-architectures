#!/usr/bin/env python3
"""Merge visually audited legacy figure decisions into the asset manifest.

The editorial audit files are intentionally kept under ``tmp/pdfs``. This
script validates them against the current README and scanner output, then
creates reproducible manifest records. It writes a preview unless ``--apply``
is passed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MANIFEST = ROOT / "assets" / "architectures" / "manifest.json"
CANDIDATES = ROOT / "tmp" / "pdfs" / "candidates-legacy.json"
PREVIEW = ROOT / "tmp" / "pdfs" / "legacy-manifest-preview.json"
AUDIT_GLOB = "editorial-audit-*.json"
OFFICIAL_ASSETS = {
    "MiniCPM-o-2.6: A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming": "https://github.com/OpenBMB/MiniCPM-o/raw/main/assets/minicpm-o-26-framework-v2.png",
    "SmolVLM: A Small, Efficient, and Open-Source Vision-Language Model": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/self_attention_architecture_smolvlm.png",
    "LLaVA 1.6: LLaVA-NeXT Improved reasoning, OCR, and world knowledge": "https://llava-vl.github.io/blog/assets/images/llava-1-6/high_res_arch_v2.png",
    "Fuyu-8B: A Multimodal Architecture for AI Agents": "https://huggingface.co/adept/fuyu-8b/resolve/main/architecture.png",
}


def normalize_figure(value: object) -> str | None:
    if value is None:
        return None
    match = re.search(r"(?:figure|fig\.?)\s*([0-9]+[a-z]?)", str(value), re.I)
    return match.group(1) if match else None


def arxiv_id_from(
    record: dict[str, object], candidate: dict[str, object]
) -> str | None:
    source = str(record.get("source_url", ""))
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", source)
    if match:
        return match.group(1)
    if not source:
        selected = candidate.get("selected_arxiv_id")
        return str(selected) if selected else None
    return None


def publication_year(arxiv_id: str | None) -> int | None:
    if not arxiv_id:
        return None
    short_year = int(arxiv_id[:2])
    return 2000 + short_year


def panel_titles() -> list[str]:
    text = README.read_text(encoding="utf-8")
    architecture = text.split("## Architectures", 1)[1].split(
        "## Important References", 1
    )[0]
    return re.findall(r"(?m)^### \*\*(.+?)\*\*\s*$", architecture)


def load_audits() -> list[dict[str, object]]:
    paths = sorted((ROOT / "tmp" / "pdfs").glob(AUDIT_GLOB))
    if not paths:
        raise RuntimeError(f"no audit files matched {AUDIT_GLOB}")
    records: list[dict[str, object]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError(f"audit must be a JSON array: {path}")
        records.extend(data)
    titles = [str(record["title"]) for record in records]
    if len(titles) != len(set(titles)):
        raise RuntimeError("duplicate titles across editorial audit files")
    return records


def matched_caption(
    candidate: dict[str, object], figure: str | None, pdf_page: object
) -> str:
    if not figure:
        return ""
    for item in candidate.get("candidates", []):
        if str(item.get("figure")) == figure and int(item.get("pdf_page", -1)) == int(
            pdf_page
        ):
            return str(item.get("caption", ""))
    return ""


def build_records() -> list[dict[str, object]]:
    titles = panel_titles()
    positions = {title: position for position, title in enumerate(titles, start=1)}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    existing_titles = {str(record["title"]) for record in manifest}
    candidate_data = json.loads(CANDIDATES.read_text(encoding="utf-8"))["records"]
    candidates = {str(record["title"]): record for record in candidate_data}
    audits = load_audits()

    legacy_titles = set(candidates)
    audited_legacy = {str(record["title"]) for record in audits} & legacy_titles
    missing = sorted(legacy_titles - audited_legacy)
    if missing:
        raise RuntimeError(f"legacy panels missing editorial decisions: {missing}")

    output: list[dict[str, object]] = []
    for audit in audits:
        title = str(audit["title"])
        if title not in legacy_titles:
            continue
        if title in existing_titles:
            raise RuntimeError(f"legacy title already exists in manifest: {title}")
        expected_position = positions.get(title)
        if expected_position != int(audit["catalog_position"]):
            raise RuntimeError(
                f"stale audit position for {title}: {audit['catalog_position']} != {expected_position}"
            )
        candidate = candidates[title]
        index = int(candidate["index"])
        status = str(audit["status"])
        base: dict[str, object] = {
            "index": index,
            "title": title,
            "slug": str(candidate["slug"]),
            "source_url": str(audit["source_url"]),
            "selection_note": str(audit["selection_note"]),
            "editorial_confidence": str(audit["confidence"]),
        }
        if status == "no-published-figure":
            output.append(
                {
                    **base,
                    "status": "no-published-figure",
                    "note": str(audit["selection_note"]),
                }
            )
            continue
        if status != "selected":
            raise RuntimeError(f"unsupported editorial status for {title}: {status}")
        arxiv_id = arxiv_id_from(audit, candidate)
        figure = normalize_figure(audit.get("figure"))
        year = publication_year(arxiv_id)
        if arxiv_id and (not figure or audit.get("pdf_page") is None):
            raise RuntimeError(f"incomplete arXiv selection for {title}")
        file_year = year or "official"
        filename = f"{candidate['slug']}-{file_year}-arch.png"
        if not arxiv_id:
            source_image_url = OFFICIAL_ASSETS.get(title)
            if not source_image_url:
                raise RuntimeError(
                    f"missing reproducible official image URL for {title}"
                )
            asset = ROOT / "assets" / "architectures" / filename
            if not asset.exists() or asset.stat().st_size < 10_000:
                raise RuntimeError(
                    f"official architecture asset is missing or too small: {asset}"
                )
        output.append(
            {
                **base,
                "status": "selected" if arxiv_id else "extracted",
                "source_kind": "arxiv" if arxiv_id else "official-web",
                "source_image_url": None if arxiv_id else source_image_url,
                "arxiv_id": arxiv_id,
                "figure": figure or str(audit["figure"]),
                "pdf_page": audit.get("pdf_page"),
                "paper_caption": matched_caption(
                    candidate, figure, audit.get("pdf_page")
                ),
                "display_caption": str(audit["display_caption"]),
                "alt": f"{title} architecture: {audit['display_caption']}",
                "file": filename,
            }
        )
    output.sort(key=lambda record: int(record["index"]))
    indices = [int(record["index"]) for record in output]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate proposed legacy manifest indices")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="append validated legacy decisions to assets/architectures/manifest.json",
    )
    args = parser.parse_args()
    records = build_records()
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"validated {len(records)} legacy decisions; preview: {PREVIEW.relative_to(ROOT)}"
    )
    if not args.apply:
        return 0
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["figures"].extend(records)
    data["figures"].sort(key=lambda record: int(record["index"]))
    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"manifest now contains {len(data['figures'])} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
