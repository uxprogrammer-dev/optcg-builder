#!/usr/bin/env python3
"""Extract One Piece TCG rule manuals into structured JSON data.

This script is intentionally lightweight so it can be run manually whenever
new rule PDFs are added.  It uses `pdfplumber` to extract text while doing
some gentle post-processing to retain page boundaries and detect section
headings heuristically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    pdfplumber = None

try:
    import pypdfium2  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    pypdfium2 = None


DATA_DIR = Path("data/manual")

DEFAULT_SOURCES = {
    "rule_manual": DATA_DIR / "rule_manual.pdf",
    "rule_comprehensive": DATA_DIR / "rule_comprehensive.pdf",
    "playsheet": DATA_DIR / "playsheet.pdf",
    "tournament_rules_manual": DATA_DIR / "tournament_rules_manual.pdf",
}


@dataclass
class Section:
    title: str
    content: List[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "paragraphs": [paragraph.strip() for paragraph in self.content if paragraph.strip()],
        }


def normalise_whitespace(text: str) -> str:
    """Collapse repeating whitespace while preserving intentional line breaks."""
    # Replace carriage returns just in case
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SECTION_HEADING_PATTERN = re.compile(r"^[A-Z0-9 \-\(\)\[\]/]{4,}$")


def split_into_sections(text: str) -> List[Section]:
    """Heuristically split large text into sections based on heading-like lines."""
    lines = text.split("\n")
    sections: List[Section] = []
    current = Section(title="Introduction", content=[])

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current.content and (not current.content[-1].endswith("\n")):
                current.content.append("")
            continue

        if SECTION_HEADING_PATTERN.match(line) and len(line) <= 120:
            if current.content:
                sections.append(current)
            current = Section(title=line.title(), content=[])
            continue

        current.content.append(line)

    if current.content:
        sections.append(current)

    return sections


def extract_with_pdfplumber(path: Path, include_pages: bool) -> tuple[str, List[dict[str, object]]]:
    if not pdfplumber:
        raise RuntimeError("pdfplumber is not available")

    with pdfplumber.open(path) as pdf:
        page_entries = []
        full_text_parts: List[str] = []

        for index, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = normalise_whitespace(raw_text)
            if cleaned:
                full_text_parts.append(cleaned)
                if include_pages:
                    page_entries.append({"page": index, "text": cleaned})

    return "\n\n".join(full_text_parts), page_entries


def extract_with_pdfium(path: Path, include_pages: bool) -> tuple[str, List[dict[str, object]]]:
    if not pypdfium2:
        raise RuntimeError("pypdfium2 is not available")

    doc = pypdfium2.PdfDocument(path)
    page_entries: List[dict[str, object]] = []
    full_text_parts: List[str] = []

    try:
        for index in range(len(doc)):
            page = doc.get_page(index)
            textpage = page.get_textpage()
            try:
                raw_text = textpage.get_text_range()
            finally:
                textpage.close()
                page.close()

            cleaned = normalise_whitespace(raw_text or "")
            if cleaned:
                full_text_parts.append(cleaned)
                if include_pages:
                    page_entries.append({"page": index + 1, "text": cleaned})
    finally:
        doc.close()

    return "\n\n".join(full_text_parts), page_entries


def extract_pdf(path: Path, *, include_pages: bool = True) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader_error: Optional[Exception] = None
    full_text = ""
    page_entries: List[dict[str, object]] = []
    page_count = 0

    if pdfplumber:
        try:
            full_text, page_entries = extract_with_pdfplumber(path, include_pages)
            # Get page count from PDF
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
        except Exception as exc:  # pragma: no cover - safety net
            loader_error = exc
            full_text = ""
            page_entries = []

    if not full_text and pypdfium2:
        try:
            full_text, page_entries = extract_with_pdfium(path, include_pages)
            # Get page count from PDF
            doc = pypdfium2.PdfDocument(path)
            try:
                page_count = len(doc)
            finally:
                doc.close()
            loader_error = None
        except Exception as exc:  # pragma: no cover - safety net
            loader_error = exc

    if not full_text and loader_error:
        raise RuntimeError(f"Failed to extract text from {path}: {loader_error}") from loader_error

    if not full_text:
        raise RuntimeError(f"No text could be extracted from {path}")

    sections = split_into_sections(full_text) if full_text else []

    return {
        "path": str(path),
        "page_count": len(page_entries) if include_pages and page_entries else page_count,
        "text": full_text,
        "pages": page_entries if include_pages else None,
        "sections": [section.to_dict() for section in sections],
    }


def build_payload(
    sources: dict[str, Path],
    *,
    include_pages: bool = True,
    summary_length: int = 8,
) -> dict[str, object]:
    payload = {
        "version": "1.0",
        "extracted_date": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "summary": "",
    }

    key_points: List[str] = []

    for key, path in sources.items():
        try:
            result = extract_pdf(path, include_pages=include_pages)
            payload["sources"][key] = {
                "path": result["path"],
                "page_count": result["page_count"],
                "sections": result["sections"],
                # Avoid storing the entire text twice when pages are included
                "text": result["text"],
            }

            # Take first sections to build quick summary bullet points
            for section in result["sections"]:
                paragraphs: Iterable[str] = section.get("paragraphs", [])
                for paragraph in paragraphs:
                    clean_para = paragraph.strip()
                    if clean_para:
                        key_points.append(f"{section['title']}: {clean_para}")
                    if len(key_points) >= summary_length:
                        break
                if len(key_points) >= summary_length:
                    break
        except Exception as exc:
            print(f"Warning: Skipping {key} ({path.name}): {exc}", file=sys.stderr)
            continue

    payload["summary"] = "\n".join(key_points[:summary_length])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract One Piece TCG rules from PDF manuals.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "rules_extracted.json",
        help="Path to write the extracted JSON payload (default: data/manual/rules_extracted.json).",
    )
    parser.add_argument(
        "--include-pages",
        action="store_true",
        help="Include per-page text dumps in the output for debugging.",
    )
    parser.add_argument(
        "--summary-length",
        type=int,
        default=12,
        help="Approximate number of key bullet points to include in the summary section.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {key: path for key, path in DEFAULT_SOURCES.items() if path.exists()}
    if not sources:
        raise SystemExit("No rule PDFs were found under data/manual/.")

    payload = build_payload(
        sources,
        include_pages=args.include_pages,
        summary_length=args.summary_length,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"Extracted rules written to {args.output} (sources: {', '.join(sources)})")


if __name__ == "__main__":
    main()

