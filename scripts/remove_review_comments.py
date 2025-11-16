#!/usr/bin/env python3
"""Remove embedded comment objects from each review in a JSON export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the `comment` object from every review in a JSON file. "
            "When --output is omitted, the input file is updated in place."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the JSON file that contains a top-level `reviews` array.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for json.dump (default: 2). Use a negative value for compact output.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding to read/write (default: utf-8).",
    )
    return parser.parse_args()


def strip_comments(payload: Dict[str, Any]) -> int:
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("JSON payload must include a list under the `reviews` key.")

    removed_count = 0
    for review in reviews:
        if isinstance(review, dict) and "comment" in review:
            review.pop("comment", None)
            removed_count += 1
    return removed_count


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path

    with input_path.open("r", encoding=args.encoding) as infile:
        payload = json.load(infile)

    removed = strip_comments(payload)

    indent = None if args.indent is not None and args.indent < 0 else args.indent
    with output_path.open("w", encoding=args.encoding) as outfile:
        json.dump(payload, outfile, ensure_ascii=False, indent=indent)
        outfile.write("\n")

    print(
        f"Removed comment objects from {removed} reviews. "
        f"Updated file: {output_path}"
    )


if __name__ == "__main__":
    main()

