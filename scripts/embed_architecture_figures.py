#!/usr/bin/env python3
"""Embed audited paper figures into the newly added architecture panels."""

from __future__ import annotations

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
    figure = html.escape(str(record["figure"]))
    page = record["pdf_page"]
    return (
        f"<!-- {marker} -->\n"
        '<p align="center">\n'
        f'  <img src="{image}" alt="{alt}" width="820">\n'
        "</p>\n"
        '<p align="center"><sub>'
        f"<b>Figure {figure}.</b> {caption} "
        f'<a href="{source}">Source paper</a>, PDF p. {page}. '
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
    sections = list(re.finditer(r"(?m)^### \*\*.+?\*\*\s*$", architecture))
    if len(sections) < len(records):
        raise RuntimeError(f"found only {len(sections)} architecture sections")

    for position in range(len(records) - 1, -1, -1):
        record = records[position]
        start = sections[position].start()
        end = sections[position + 1].start() if position + 1 < len(sections) else len(architecture)
        section = architecture[start:end]
        marker_pattern = re.compile(
            rf"<!-- architecture-figure:{record['index']} -->.*?"
            rf"<!-- /architecture-figure:{record['index']} -->\n*",
            re.S,
        )
        section = marker_pattern.sub("", section)
        details = re.search(r"(?m)^<details>\s*$", section)
        if not details:
            raise RuntimeError(f"missing details panel for entry {record['index']}")
        before = section[: details.start()].rstrip("\n")
        remainder = section[details.start() :].lstrip("\n")
        section = before + "\n\n" + figure_block(record) + remainder
        architecture = architecture[:start] + section + architecture[end:]

    README.write_text(prefix + architecture + suffix, encoding="utf-8")


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
        source = f"[paper]({record['source_url']})"
        figure = f"Fig. {record['figure']}, PDF p. {record['pdf_page']}"
        local = f"[{record['file']}]({record['file']})"
        lines.append(f"| {title} | {source} | {figure} | {local} |")
    lines.append("")
    CREDITS.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    update_readme(records)
    write_credits(records)
    print(f"embedded {sum(r['status'] == 'extracted' for r in records)} figures")
    print(f"documented {sum(r['status'] == 'no-published-figure' for r in records)} no-figure sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
