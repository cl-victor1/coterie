#!/usr/bin/env python3
"""
Utility script to download all available reviews for Coterie's "The Diaper"
product by paging through the Yotpo-backed API that powers the PDP.

Example:
    python scripts/fetch_coterie_reviews.py --output output/the_diaper_reviews.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

BASE_URL = "https://www.coterie.com/api/yotpo"
DEFAULT_PRODUCT_ID = "4471557914690"
MAX_PER_PAGE = 100
REQUEST_TIMEOUT = 25
RETRY_ATTEMPTS = 3
BACKOFF_SECONDS = 1.5
USER_AGENT = "codex-coterie-review-scraper/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Yotpo reviews for https://www.coterie.com/products/the-diaper"
    )
    parser.add_argument(
        "--product-id",
        default=DEFAULT_PRODUCT_ID,
        help="Yotpo product ID. Default corresponds to The Diaper.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=MAX_PER_PAGE,
        help="Number of reviews per API call (max observed 100).",
    )
    parser.add_argument(
        "--sort",
        default="recent",
        help="Sort order passed to the Yotpo search endpoint (for example 'recent' or 'highest_score').",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional upper bound on pages to fetch (useful for quick tests).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests to be polite.",
    )
    parser.add_argument(
        "--output",
        default="output/the_diaper_reviews.json",
        help="Destination file path. Directories are created automatically.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Serialization format for the output file.",
    )
    parser.add_argument(
        "--filters-file",
        help="Optional path to a JSON file containing extra filters passed to the search endpoint.",
    )
    return parser.parse_args()


def load_filters(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    filters_path = Path(path)
    if not filters_path.exists():
        raise FileNotFoundError(f"Filters file not found: {filters_path}")
    with filters_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = session.request(
                method,
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            sleep_for = BACKOFF_SECONDS * attempt
            print(
                f"[retry {attempt}/{RETRY_ATTEMPTS}] {error}; sleeping {sleep_for:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"Exhausted retries for {url}") from last_error


def fetch_bottom_line(session: requests.Session, product_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/{product_id}/bottom-line"
    return request_json(session, "GET", url, payload=None)


def fetch_reviews_page(
    session: requests.Session,
    product_id: str,
    page: int,
    per_page: int,
    *,
    sort: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    per_page = min(max(per_page, 1), MAX_PER_PAGE)
    url = f"{BASE_URL}/{product_id}/search-reviews"
    payload: Dict[str, Any] = {"page": page, "perPage": per_page}
    if sort:
        payload["sort"] = sort
    if filters:
        payload["filters"] = filters
    return request_json(session, "POST", url, payload=payload)


def flatten_review(review: Dict[str, Any]) -> Dict[str, Any]:
    user = review.get("user") or {}
    comment = review.get("comment") or {}
    flattened = {
        "id": review.get("id"),
        "score": review.get("score"),
        "title": review.get("title"),
        "content": review.get("content"),
        "created_at": review.get("created_at"),
        "verified_buyer": review.get("verified_buyer"),
        "votes_up": review.get("votes_up"),
        "votes_down": review.get("votes_down"),
        "product_id": review.get("product_id"),
        "user_id": user.get("user_id"),
        "user_display_name": user.get("display_name"),
        "user_type": user.get("user_type"),
        "user_is_social_connected": user.get("is_social_connected"),
        "comment_id": comment.get("id"),
        "comment_display_name": comment.get("display_name"),
        "comment_created_at": comment.get("created_at"),
        "comment_content": comment.get("content"),
        "custom_fields": json.dumps(review.get("custom_fields", {}), ensure_ascii=False),
    }
    return flattened


def write_json(output_path: Path, metadata: Dict[str, Any], reviews: List[Dict[str, Any]]) -> None:
    payload = {"metadata": metadata, "reviews": reviews}
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(output_path: Path, metadata: Dict[str, Any], reviews: List[Dict[str, Any]]) -> None:
    fieldnames = list(flatten_review(reviews[0]).keys()) if reviews else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            writer.writerow(flatten_review(review))
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def iter_pages(
    session: requests.Session,
    product_id: str,
    per_page: int,
    *,
    sort: str,
    filters: Optional[Dict[str, Any]],
    delay: float,
    max_pages: Optional[int],
) -> Iterable[Dict[str, Any]]:
    page = 1
    while True:
        yield fetch_reviews_page(
            session,
            product_id,
            page,
            per_page,
            sort=sort,
            filters=filters,
        )
        page += 1
        if max_pages and page > max_pages:
            break
        if delay:
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    filters = load_filters(args.filters_file)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    per_page = min(max(args.per_page, 1), MAX_PER_PAGE)
    bottom_line = fetch_bottom_line(session, args.product_id)
    expected_total = bottom_line.get("reviewsCount")

    print(
        f"Bottom-line stats: score={bottom_line.get('score')} reviews={expected_total}",
        file=sys.stderr,
    )

    reviews: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    total_from_pages = 0

    for page_index, page_data in enumerate(
        iter_pages(
            session,
            args.product_id,
            per_page,
            sort=args.sort,
            filters=filters,
            delay=args.delay,
            max_pages=args.max_pages,
        ),
        start=1,
    ):
        batch = page_data.get("reviews", [])
        pagination = page_data.get("pagination", {})
        total_from_pages = max(total_from_pages, pagination.get("total", 0))
        if not batch:
            print("No more reviews returned; stopping pagination.", file=sys.stderr)
            break

        for review in batch:
            review_id = review.get("id")
            if review_id in seen_ids:
                continue
            reviews.append(review)
            if review_id is not None:
                seen_ids.add(review_id)

        print(
            f"Fetched page {page_index}: {len(batch)} reviews "
            f"(cumulative {len(reviews)}/{expected_total or total_from_pages})",
            file=sys.stderr,
        )

        target = expected_total or total_from_pages
        if target and len(reviews) >= target:
            break

    if expected_total and len(reviews) < expected_total:
        print(
            f"Warning: expected {expected_total} reviews but only downloaded {len(reviews)}. "
            "This can happen if some reviews are hidden from the search API.",
            file=sys.stderr,
        )

    if not reviews:
        raise RuntimeError("No reviews were downloaded; aborting.")

    fetched_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    metadata = {
        "product_id": args.product_id,
        "per_page": per_page,
        "sort": args.sort,
        "filters": filters,
        "expected_reviews_from_bottom_line": expected_total,
        "expected_reviews_from_pagination": total_from_pages,
        "downloaded_reviews": len(reviews),
        "fetched_at": fetched_at,
        "source": f"{BASE_URL}/{args.product_id}/search-reviews",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        write_json(output_path, metadata, reviews)
    else:
        write_csv(output_path, metadata, reviews)

    print(
        f"Saved {len(reviews)} reviews to {output_path} (format={args.format}).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
