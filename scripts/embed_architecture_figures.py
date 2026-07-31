#!/usr/bin/env python3
"""Embed audited paper figures into the newly added architecture panels."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MANIFEST = ROOT / "assets" / "architectures" / "manifest.json"
CREDITS = ROOT / "assets" / "architectures" / "CREDITS.md"


def figure_block(record: dict[str, object]) -> str:
    index = record["index"]
    marker = f"architecture-figure:{index}"
    if record["status"] == "no-published-figure":
        note = html.escape(str(record["note"]))
        return (
            f"<!-- {marker} -->\n"
            f"> **Architecture figure:** {note}\n"
            f"<!-- /{marker} -->\n\n"
        )

    source = html.escape(str(record["source_url"]), quote=True)
    alt = html.escape(str(record["alt"]), quote=True)
    caption = html.escape(str(record["display_caption"]))
    image = html.escape(f"assets/architectures/{record['file']}", quote=True)
    source_kind = str(record.get("source_kind", "arxiv"))
    if source_kind == "official-web":
        credit = (
            f"<b>Official architecture diagram.</b> {caption} "
            f'<a href="{source}">Primary source</a>. '
        )
    else:
        figure = html.escape(str(record["figure"]))
        page = record["pdf_page"]
        credit = (
            f"<b>Figure {figure}.</b> {caption} "
            f'<a href="{source}">Source paper</a>, PDF p. {page}. '
        )
    return (
        f"<!-- {marker} -->\n"
        '<p align="center">\n'
        f'  <img src="{image}" alt="{alt}" width="820">\n'
        "</p>\n"
        '<p align="center"><sub>'
        f"{credit}"
        '<a href="assets/architectures/FIGURE_NOTICE.md">Figure notice</a>.'
        "</sub></p>\n"
        f"<!-- /{marker} -->\n\n"
    )


def update_readme(records: list[dict[str, object]]) -> None:
    text = README.read_text(encoding="utf-8")
    architecture_start = text.index("## Architectures")
    important_start = text.index("## Important References", architecture_start)
    prefix, architecture, suffix = (
        text[:architecture_start],
        text[architecture_start:important_start],
        text[important_start:],
    )
    matches = list(re.finditer(r"(?m)^### \*\*(.+?)\*\*\s*$", architecture))
    sections = [
        architecture[
            match.start() : matches[position + 1].start()
            if position + 1 < len(matches)
            else len(architecture)
        ]
        for position, match in enumerate(matches)
    ]
    titles = [match.group(1) for match in matches]
    if len(titles) != len(set(titles)):
        raise RuntimeError(
            "duplicate architecture titles prevent stable figure matching"
        )
    positions = {title: position for position, title in enumerate(titles)}

    for record in records:
        title = str(record["title"])
        if title not in positions:
            raise RuntimeError(f"missing architecture panel for figure record: {title}")
        position = positions[title]
        section = sections[position]
        marker_pattern = re.compile(
            rf"<!-- architecture-figure:{record['index']} -->.*?"
            rf"<!-- /architecture-figure:{record['index']} -->\n*",
            re.S,
        )
        section = marker_pattern.sub("", section)
        # Legacy panels used mutable user-attachment images without durable
        # figure provenance. Remove that entire centered block before adding
        # the audited local asset (or an explicit no-figure decision).
        section = re.sub(
            r'<p\s+align="center">\s*<img\s+src="https?://[^>]*?/?>\s*(?:</p>\s*)?',
            "",
            section,
            flags=re.I,
        )
        section = re.sub(
            r'<p\s+align="center">\s*<img\b.*?</p>\s*',
            "",
            section,
            flags=re.I | re.S,
        )
        details = re.search(r"(?m)^<details\b[^>]*>", section)
        if not details:
            raise RuntimeError(f"missing details panel for entry {record['index']}")
        before = section[: details.start()].rstrip("\n")
        remainder = section[details.start() :].lstrip("\n")
        sections[position] = before + "\n\n" + figure_block(record) + remainder

    leading = architecture[: matches[0].start()]
    README.write_text(prefix + leading + "".join(sections) + suffix, encoding="utf-8")


def write_credits(records: list[dict[str, object]]) -> None:
    lines = [
        "# Architecture figure credits",
        "",
        "Each image is an excerpt from the linked primary source. Copyright remains with the paper authors or publishers; these excerpts are not covered by the repository's license. See the [figure notice](FIGURE_NOTICE.md).",
        "",
        "| Entry | Source | Figure | Local file |",
        "| --- | --- | ---: | --- |",
    ]
    for record in records:
        if record["status"] != "extracted":
            continue
        title = str(record["title"]).replace("|", "\\|")
        source_kind = str(record.get("source_kind", "arxiv"))
        source = f"[primary source]({record['source_url']})"
        figure = (
            "Official architecture diagram"
            if source_kind == "official-web"
            else f"Fig. {record['figure']}, PDF p. {record['pdf_page']}"
        )
        local = f"[{record['file']}]({record['file']})"
        lines.append(f"| {title} | {source} | {figure} | {local} |")
    lines.append("")
    CREDITS.write_text("\n".join(lines), encoding="utf-8")


def verify_readme(records: list[dict[str, object]]) -> None:
    text = README.read_text(encoding="utf-8")
    architecture = text.split("## Architectures", 1)[1].split(
        "## Important References", 1
    )[0]
    matches = list(re.finditer(r"(?m)^### \*\*(.+?)\*\*\s*$", architecture))
    sections = {
        match.group(1): architecture[
            match.start() : matches[position + 1].start()
            if position + 1 < len(matches)
            else len(architecture)
        ]
        for position, match in enumerate(matches)
    }
    titles = [str(record["title"]) for record in records]
    indices = [int(record["index"]) for record in records]
    slugs = [str(record["slug"]) for record in records]
    for name, values in (("title", titles), ("index", indices), ("slug", slugs)):
        if len(values) != len(set(values)):
            raise RuntimeError(f"duplicate figure manifest {name}")
    for record in records:
        title = str(record["title"])
        if title not in sections:
            raise RuntimeError(f"missing architecture panel for figure record: {title}")
        start_marker = f"<!-- architecture-figure:{record['index']} -->"
        end_marker = f"<!-- /architecture-figure:{record['index']} -->"
        section = sections[title]
        if section.count(start_marker) != 1 or section.count(end_marker) != 1:
            raise RuntimeError(f"figure markers are not attached to {title}")
        if text.count(start_marker) != 1 or text.count(end_marker) != 1:
            raise RuntimeError(f"figure markers are duplicated for {title}")
        if record["status"] == "extracted":
            asset = ROOT / "assets" / "architectures" / str(record["file"])
            if not asset.exists() or asset.stat().st_size < 10_000:
                raise RuntimeError(f"missing or undersized architecture asset: {asset}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify stable figure-to-panel mappings without rewriting files",
    )
    args = parser.parse_args()
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    if args.check:
        verify_readme(records)
        print(f"architecture figure mapping is current: {len(records)}/{len(records)}")
        return 0
    update_readme(records)
    write_credits(records)
    print(f"embedded {sum(r['status'] == 'extracted' for r in records)} figures")
    print(
        f"documented {sum(r['status'] == 'no-published-figure' for r in records)} no-figure sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
