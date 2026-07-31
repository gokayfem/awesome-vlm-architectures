#!/usr/bin/env python3
"""Render and crop architecture figures selected in the asset manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import fitz
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "architectures" / "manifest.json"
CROP_OVERRIDES = ROOT / "assets" / "architectures" / "crop_overrides.json"
PDF_DIR = ROOT / "tmp" / "pdfs"
OUTPUT_DIR = ROOT / "assets" / "architectures"
CONTACT_DIR = PDF_DIR / "contacts"
RENDER_DPI = 360
MAX_WIDTH = 1800


def ensure_pdf(arxiv_id: str) -> Path:
    """Return a cached primary-source PDF, downloading it when necessary."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{arxiv_id}.pdf"
    if pdf.exists() and pdf.stat().st_size >= 50_000:
        return pdf
    temporary = pdf.with_suffix(".pdf.part")
    request = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": "awesome-vlm-architectures/figure-extractor"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        temporary.write_bytes(response.read())
    temporary.replace(pdf)
    time.sleep(0.15)
    return pdf


def normalize_figure(value: str) -> str:
    return re.sub(
        r"[^0-9a-z]", "", value.lower().replace("figure", "").replace("fig", "")
    )


def find_caption_block(page: fitz.Page, figure: str) -> tuple[fitz.Rect, str]:
    wanted = normalize_figure(figure)
    patterns = [
        re.compile(rf"(?:Figure|Fig\.)\s*{re.escape(wanted)}\b", re.I),
        re.compile(
            rf"(?:Figure|Fig\.)\s*{re.escape(wanted.rstrip('abcdefghijklmnopqrstuvwxyz'))}\b",
            re.I,
        ),
    ]
    matches: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("blocks", sort=True):
        text = re.sub(r"\s+", " ", block[4]).strip()
        if any(pattern.search(text) for pattern in patterns):
            matches.append((fitz.Rect(block[:4]), text))
    if not matches:
        raise ValueError(
            f"caption for {figure} not found on PDF page {page.number + 1}"
        )
    matches.sort(
        key=lambda item: (
            0 if item[1].lower().lstrip().startswith(("figure", "fig.")) else 1,
            item[0].y0,
        )
    )
    return matches[0]


def prose_block_before(
    page: fitz.Page, caption: fitz.Rect, x_clip: fitz.Rect
) -> fitz.Rect | None:
    candidates = []
    for block in page.get_text("blocks", sort=True):
        rect = fitz.Rect(block[:4])
        text = re.sub(r"\s+", " ", block[4]).strip()
        overlap = max(0.0, min(rect.x1, x_clip.x1) - max(rect.x0, x_clip.x0))
        if (
            rect.y1 < caption.y0 - 14
            and overlap > min(rect.width, x_clip.width) * 0.45
            and len(text) > 180
            and len(text.split()) > 28
        ):
            candidates.append(rect)
    return max(candidates, key=lambda rect: rect.y1) if candidates else None


def initial_clip(page: fitz.Page, caption: fitz.Rect) -> fitz.Rect:
    page_rect = page.rect
    if caption.width < page_rect.width * 0.57:
        center = (caption.x0 + caption.x1) / 2
        if center < page_rect.width / 2:
            x0, x1 = page_rect.width * 0.035, page_rect.width * 0.51
        else:
            x0, x1 = page_rect.width * 0.49, page_rect.width * 0.965
    else:
        x0, x1 = page_rect.width * 0.035, page_rect.width * 0.965
    horizontal = fitz.Rect(x0, 0, x1, page_rect.height)
    previous = prose_block_before(page, caption, horizontal)
    top = previous.y1 + 5 if previous else max(page_rect.y0 + 32, caption.y0 - 390)
    if caption.y0 - top < 88:
        top = max(page_rect.y0 + 28, caption.y0 - 230)
    if caption.y0 - top > 420:
        top = caption.y0 - 420
    return fitz.Rect(x0, top, x1, max(top + 20, caption.y0 - 3))


def trim_white(image: Image.Image, padding: int = 18) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 12 else 0)
    bbox = difference.getbbox()
    if not bbox:
        return rgb
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(rgb.width, bbox[2] + padding)
    bottom = min(rgb.height, bbox[3] + padding)
    return rgb.crop((left, top, right, bottom))


def extract(record: dict[str, object]) -> Path:
    arxiv_id = str(record["arxiv_id"])
    page_number = int(record["pdf_page"])
    figure = str(record["figure"])
    pdf = ensure_pdf(arxiv_id)
    document = fitz.open(pdf)
    if not 1 <= page_number <= len(document):
        raise ValueError(
            f"invalid page {page_number} for {arxiv_id} ({len(document)} pages)"
        )
    page = document[page_number - 1]
    caption_rect, caption_text = find_caption_block(page, figure)
    override = record.get("crop_override_pdf_points")
    clip = fitz.Rect(override) if override else initial_clip(page, caption_rect)
    zoom = RENDER_DPI / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    image = trim_white(image)
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height), Image.Resampling.LANCZOS)
    destination = OUTPUT_DIR / str(record["file"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True, compress_level=9)
    record["extracted_caption"] = caption_text
    record["crop_pdf_points"] = [round(value, 2) for value in clip]
    record["image_size"] = [image.width, image.height]
    return destination


def make_contact_sheet(records: list[dict[str, object]], group: int) -> Path:
    selected = [record for record in records if record.get("status") == "extracted"]
    selected = selected[(group - 1) * 12 : group * 12]
    if not selected:
        raise ValueError(f"no records for contact sheet {group}")
    thumb_width, cell_height = 620, 470
    columns = 2
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * cell_height), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for position, record in enumerate(selected):
        image = Image.open(OUTPUT_DIR / str(record["file"])).convert("RGB")
        image.thumbnail((thumb_width - 24, cell_height - 58), Image.Resampling.LANCZOS)
        x = (position % columns) * thumb_width + (thumb_width - image.width) // 2
        y = (
            (position // columns) * cell_height
            + 38
            + (cell_height - 48 - image.height) // 2
        )
        sheet.paste(image, (x, y))
        label = f"{record['index']:02} {record['slug']} - Fig {record['figure']} p{record['pdf_page']}"
        draw.text(
            (
                (position % columns) * thumb_width + 10,
                (position // columns) * cell_height + 9,
            ),
            label,
            fill="black",
            font=font,
        )
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = CONTACT_DIR / f"contact-{group:02}.jpg"
    sheet.save(destination, quality=88, optimize=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contacts", action="store_true", help="also create 12-up QA contact sheets"
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="extract only records with selected status, leaving reviewed assets untouched",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        help="extract only these stable manifest indices, regardless of current status",
    )
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    overrides = json.loads(CROP_OVERRIDES.read_text(encoding="utf-8"))[
        "pdf_points_by_manifest_index"
    ]
    for record in data["figures"]:
        override = overrides.get(str(record["index"]))
        if override:
            record["crop_override_pdf_points"] = override
    failures = []
    for record in data["figures"]:
        allowed_statuses = {"selected"} if args.new_only else {"selected", "extracted"}
        if args.indices and int(record["index"]) not in args.indices:
            continue
        if record.get("status") not in allowed_statuses:
            continue
        try:
            destination = extract(record)
            record["status"] = "extracted"
            print(f"[{record['index']:02}] {destination.relative_to(ROOT)}")
        except Exception as error:  # noqa: BLE001 - report all batch failures
            record["status"] = "failed"
            record["error"] = str(error)
            failures.append((record["index"], str(error)))
            print(f"[{record['index']:02}] FAILED: {error}")
    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.contacts:
        count = sum(record.get("status") == "extracted" for record in data["figures"])
        for group in range(1, math.ceil(count / 12) + 1):
            print(make_contact_sheet(data["figures"], group).relative_to(ROOT))
    if failures:
        print(f"{len(failures)} extraction failure(s): {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
