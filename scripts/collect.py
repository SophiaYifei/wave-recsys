"""Data collection across modalities (books, films, music, writing).

Usage:
    python scripts/collect.py --source {books,films,music,writing} --target-count N

Each sub-collector writes to data/raw/{source}/raw.jsonl in append-only mode
and is resumable: items already present in the output are skipped on rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def output_path(source: str) -> Path:
    """Return the raw.jsonl path for a given source."""
    path = RAW_DIR / source / "raw.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_existing_ids(path: Path) -> Set[str]:
    """Read every id already written to a jsonl file (for resumable runs)."""
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    """Append one JSON record (one line) to a jsonl file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def http_get_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    timeout: int = 20,
) -> Optional[requests.Response]:
    """GET with exponential backoff. Returns None if all retries fail."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                # Rate limited — back off harder
                sleep_for = (2 ** attempt) * 5
                time.sleep(sleep_for)
                continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return resp
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Books — goodbooks-10k + Open Library
# ---------------------------------------------------------------------------

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPEN_LIBRARY_WORK = "https://openlibrary.org{work_key}.json"
BOOKS_CSV = RAW_DIR / "books" / "books.csv"


def _extract_description(work_payload: Dict[str, Any]) -> str:
    """Open Library 'description' may be a string OR {value: string} OR missing."""
    desc = work_payload.get("description")
    if isinstance(desc, str):
        return desc.strip()
    if isinstance(desc, dict):
        return str(desc.get("value", "")).strip()
    return ""


def _first_subject(work_payload: Dict[str, Any]) -> str:
    """Pick a single subject as 'genre' surrogate (Open Library has no formal genre)."""
    subjects = work_payload.get("subjects") or []
    return str(subjects[0]) if subjects else ""


def _fetch_open_library(title: str, author: str, sleep_seconds: float) -> Dict[str, Any]:
    """Search Open Library for a work matching (title, author).

    Returns a dict with keys 'description' (str) and 'genre' (str). Either may be empty.
    Sleeps `sleep_seconds` between the two API calls to stay polite.
    """
    out = {"description": "", "genre": ""}
    search = http_get_with_retry(
        OPEN_LIBRARY_SEARCH,
        params={"title": title, "author": author, "limit": 1},
    )
    if search is None or search.status_code != 200:
        return out
    docs = search.json().get("docs") or []
    if not docs:
        return out
    work_key = docs[0].get("key")
    if not work_key:
        return out

    time.sleep(sleep_seconds)
    work = http_get_with_retry(OPEN_LIBRARY_WORK.format(work_key=work_key))
    if work is None or work.status_code != 200:
        return out
    payload = work.json()
    out["description"] = _extract_description(payload)
    out["genre"] = _first_subject(payload)
    return out


def collect_books(target_count: int, sleep_seconds: float = 0.5) -> None:
    """Build data/raw/books/raw.jsonl by enriching goodbooks-10k with Open Library."""
    if not BOOKS_CSV.exists():
        print(
            f"ERROR: {BOOKS_CSV} not found.\n"
            "Download it manually:\n"
            "  curl -L https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv "
            f"-o {BOOKS_CSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = output_path("books")
    existing_ids = load_existing_ids(out_path)
    print(f"books: {len(existing_ids)} items already present in {out_path}")

    rows: List[Dict[str, str]] = []
    with BOOKS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _ratings_count(row: Dict[str, str]) -> int:
        try:
            return int(row.get("ratings_count") or 0)
        except ValueError:
            return 0

    rows.sort(key=_ratings_count, reverse=True)
    rows = rows[:target_count]

    max_ratings = max((_ratings_count(r) for r in rows), default=1) or 1

    written = 0
    skipped = 0
    for idx, row in enumerate(tqdm(rows, desc="books"), start=1):
        item_id = f"book_{idx:04d}"
        if item_id in existing_ids:
            skipped += 1
            continue

        title = (row.get("original_title") or row.get("title") or "").strip()
        display_title = (row.get("title") or title).strip()
        authors = (row.get("authors") or "").strip()
        try:
            year = int(float(row.get("original_publication_year") or 0))
        except ValueError:
            year = 0
        ratings = _ratings_count(row)
        cover_url = (row.get("image_url") or "").strip()
        gr_id = (row.get("goodreads_book_id") or "").strip()
        external_url = (
            f"https://www.goodreads.com/book/show/{gr_id}" if gr_id else ""
        )

        enrichment = {"description": "", "genre": ""}
        if title and authors:
            enrichment = _fetch_open_library(title, authors, sleep_seconds)
            time.sleep(sleep_seconds)

        description = enrichment["description"] or f"A book by {authors} ({year})."

        item = {
            "id": item_id,
            "modality": "book",
            "title": display_title,
            "creator": authors,
            "year": year,
            "description": description,
            "reviews": [],
            "popularity_score": ratings / max_ratings if max_ratings else 0.0,
            "cover_url": cover_url,
            "external_url": external_url,
            "modality_specific": {
                "genre": enrichment["genre"],
                "page_count": 0,
            },
        }
        append_jsonl(out_path, item)
        written += 1

        if (idx % 50) == 0:
            print(
                f"  progress: idx={idx}/{len(rows)} written={written} skipped={skipped}",
                flush=True,
            )

    print(f"books: done. wrote {written} new items, skipped {skipped} existing.")


# ---------------------------------------------------------------------------
# Films — TMDB top-rated movies + TV
# ---------------------------------------------------------------------------

TMDB_BASE = "https://api.themoviedb.org/3"


def _tmdb_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Hit a TMDB endpoint with API key + retry. Returns parsed JSON or None."""
    api_key = os.environ.get("TMDB_API_KEY")
    full_params = {"api_key": api_key, "language": "en-US"}
    if params:
        full_params.update(params)
    resp = http_get_with_retry(f"{TMDB_BASE}{path}", params=full_params)
    if resp is None or resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _fetch_top_rated_ids(kind: str, count: int) -> List[int]:
    """kind in {'movie', 'tv'}. Page through top_rated until we have `count` ids."""
    ids: List[int] = []
    page = 1
    while len(ids) < count:
        data = _tmdb_get(f"/{kind}/top_rated", {"page": page})
        if not data or not data.get("results"):
            break
        for r in data["results"]:
            ids.append(int(r["id"]))
            if len(ids) >= count:
                break
        if page >= int(data.get("total_pages", 1)):
            break
        page += 1
    return ids[:count]


def _build_film_item(kind: str, tmdb_id: int) -> Optional[Dict[str, Any]]:
    """Fetch detail + reviews for one TMDB item; returns a partial record.

    The 'id' and normalized popularity are filled in by the caller.
    """
    detail = _tmdb_get(f"/{kind}/{tmdb_id}")
    if not detail:
        return None

    reviews_data = _tmdb_get(f"/{kind}/{tmdb_id}/reviews")
    reviews: List[str] = []
    if reviews_data and reviews_data.get("results"):
        for r in reviews_data["results"][:3]:
            content = (r.get("content") or "").strip()
            if content:
                reviews.append(content[:500])

    is_tv = kind == "tv"
    title = detail.get("name") if is_tv else detail.get("title")
    date_field = "first_air_date" if is_tv else "release_date"
    year_str = (detail.get(date_field) or "").split("-")[0]
    try:
        year = int(year_str) if year_str else 0
    except ValueError:
        year = 0

    description = (detail.get("overview") or "").strip() or (
        f"A TV series ({year})." if is_tv else f"A film ({year})."
    )

    companies = detail.get("production_companies") or []
    creator = ", ".join(c.get("name", "") for c in companies[:2] if c.get("name"))

    poster = detail.get("poster_path") or ""
    cover_url = f"https://image.tmdb.org/t/p/w500{poster}" if poster else ""
    external_url = f"https://www.themoviedb.org/{kind}/{tmdb_id}"

    genres = detail.get("genres") or []
    genre = ", ".join(g.get("name", "") for g in genres[:3] if g.get("name"))

    if is_tv:
        run_times = detail.get("episode_run_time") or []
        runtime = int(run_times[0]) if run_times else 0
    else:
        runtime = int(detail.get("runtime") or 0)

    pop = float(detail.get("popularity") or 0.0)

    return {
        "_popularity_raw": pop,
        "modality": "film",
        "title": title or "",
        "creator": creator,
        "year": year,
        "description": description,
        "reviews": reviews,
        "cover_url": cover_url,
        "external_url": external_url,
        "modality_specific": {
            "genre": genre,
            "runtime": runtime,
            "is_tv": is_tv,
        },
    }


def collect_films(target_count: int = 600) -> None:
    """Build data/raw/films/raw.jsonl from TMDB top-rated movies + TV (2:1 split)."""
    if not os.environ.get("TMDB_API_KEY"):
        print("ERROR: TMDB_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    out_path = output_path("films")
    existing_ids = load_existing_ids(out_path)
    print(f"films: {len(existing_ids)} items already present in {out_path}")

    movies_count = (target_count * 2) // 3
    tv_count = target_count - movies_count
    print(f"films: fetching top-rated lists ({movies_count} movie + {tv_count} tv)")
    movie_ids = _fetch_top_rated_ids("movie", movies_count)
    tv_ids = _fetch_top_rated_ids("tv", tv_count)
    print(f"films: got {len(movie_ids)} movie + {len(tv_ids)} tv ids from TMDB")

    plan: List[Tuple[int, str, int]] = []
    for i, mid in enumerate(movie_ids, start=1):
        plan.append((i, "movie", mid))
    for j, tid in enumerate(tv_ids, start=len(movie_ids) + 1):
        plan.append((j, "tv", tid))

    todo = [(idx, kind, tid) for idx, kind, tid in plan if f"film_{idx:04d}" not in existing_ids]
    print(f"films: {len(todo)} items to fetch ({len(plan) - len(todo)} already done)")

    items: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_build_film_item, k, t): (idx, k, t) for idx, k, t in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="films"):
            idx, kind, tmdb_id = futures[fut]
            try:
                item = fut.result()
            except Exception as exc:
                print(f"  failed film_{idx:04d} ({kind}/{tmdb_id}): {exc}", file=sys.stderr)
                continue
            if item is None:
                print(f"  skipped film_{idx:04d} ({kind}/{tmdb_id}): no detail", file=sys.stderr)
                continue
            item["id"] = f"film_{idx:04d}"
            items.append(item)

    if not items:
        print("films: nothing new to write.")
        return

    max_pop = max(it["_popularity_raw"] for it in items) or 1.0
    items.sort(key=lambda it: it["id"])
    for item in items:
        record = {
            "id": item["id"],
            "modality": "film",
            "title": item["title"],
            "creator": item["creator"],
            "year": item["year"],
            "description": item["description"],
            "reviews": item["reviews"],
            "popularity_score": item.pop("_popularity_raw") / max_pop,
            "cover_url": item["cover_url"],
            "external_url": item["external_url"],
            "modality_specific": item["modality_specific"],
        }
        append_jsonl(out_path, record)

    print(f"films: wrote {len(items)} new items")


# ---------------------------------------------------------------------------
# Writing — Longform RSS + New Yorker Fiction RSS + Poetry Foundation
# ---------------------------------------------------------------------------

LONGFORM_RSS = "https://longform.org/feed"
NEW_YORKER_FICTION_RSS = "https://www.newyorker.com/feed/magazine/fiction"
POETRY_FOUNDATION_BROWSE = "https://www.poetryfoundation.org/poems/browse"
USER_AGENT = "wave-recsys/0.1 (academic project; contact: cnguoyifei@gmail.com)"


def _strip_html(html_text: str) -> str:
    """Convert HTML to plain text with collapsed whitespace."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _entry_year(entry: Any) -> int:
    """Pull a year from a feedparser entry's published/updated date."""
    for key in ("published", "updated", "created"):
        value = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
        if not value:
            continue
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(value).year
        except (TypeError, ValueError):
            continue
    return 0


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _fetch_rss_items(
    rss_url: str,
    target: int,
    item_type: str,
    publication: str,
    fallback_creator: str = "",
) -> List[Dict[str, Any]]:
    """Generic RSS fetch returning normalized writing items (no id, no popularity)."""
    feed = feedparser.parse(rss_url, agent=USER_AGENT)
    items: List[Dict[str, Any]] = []
    for entry in feed.entries:
        if len(items) >= target:
            break
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        creator = (entry.get("author") or "").strip() or fallback_creator
        link = entry.get("link") or ""
        summary = _strip_html(entry.get("summary") or entry.get("description") or "")
        description = summary[:500] or f"A {item_type} by {creator or 'unknown'}."
        items.append(
            {
                "modality": "writing",
                "title": title,
                "creator": creator,
                "year": _entry_year(entry),
                "description": description,
                "reviews": [],
                "cover_url": "",
                "external_url": link,
                "modality_specific": {
                    "type": item_type,
                    "word_count": _word_count(summary),
                    "publication": publication,
                },
            }
        )
    return items


def _fetch_poetry_foundation(target: int) -> List[Dict[str, Any]]:
    """Scrape Poetry Foundation's 'browse' listing sorted by popularity (lifetime views)."""
    headers = {"User-Agent": USER_AGENT}
    items: List[Dict[str, Any]] = []
    page = 1
    while len(items) < target:
        resp = http_get_with_retry(
            POETRY_FOUNDATION_BROWSE,
            params={"sort_by": "popular_views_lifetime", "page": page},
            headers=headers,
        )
        if resp is None or resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        # Poem cards live in <article> elements on the browse page.
        articles = soup.find_all("article")
        if not articles:
            break
        page_items = 0
        for art in articles:
            if len(items) >= target:
                break
            title_link = art.find("a", href=lambda h: h and "/poems/" in h)
            if not title_link:
                continue
            title = title_link.get_text(strip=True)
            href = title_link["href"]
            if href.startswith("/"):
                href = "https://www.poetryfoundation.org" + href
            # Author often in a sibling link
            author_link = art.find("a", href=lambda h: h and "/poets/" in h)
            creator = author_link.get_text(strip=True) if author_link else ""
            # Excerpt: longest text block in the article excluding the title
            excerpt = ""
            for p in art.find_all(["p", "div"]):
                txt = p.get_text(" ", strip=True)
                if txt and txt != title and len(txt) > len(excerpt):
                    excerpt = txt
            if not excerpt:
                excerpt = f"A poem by {creator or 'unknown'}."
            items.append(
                {
                    "modality": "writing",
                    "title": title,
                    "creator": creator,
                    "year": 0,
                    "description": excerpt[:500],
                    "reviews": [],
                    "cover_url": "",
                    "external_url": href,
                    "modality_specific": {
                        "type": "poem",
                        "word_count": _word_count(excerpt),
                        "publication": "Poetry Foundation",
                    },
                }
            )
            page_items += 1
        if page_items == 0:
            break
        page += 1
        time.sleep(1.0)
    return items


def collect_writing(target_count: int = 1300) -> None:
    """Build data/raw/writing/raw.jsonl from Longform + New Yorker Fiction + Poetry Foundation."""
    out_path = output_path("writing")
    existing_ids = load_existing_ids(out_path)
    print(f"writing: {len(existing_ids)} items already present in {out_path}")

    # Spec target split: Longform 600 / Poetry 400 / NY Fiction 300
    longform_target = (target_count * 600) // 1300
    poetry_target = (target_count * 400) // 1300
    nyf_target = target_count - longform_target - poetry_target

    print(f"writing: fetching Longform RSS (target {longform_target})")
    longform_items = _fetch_rss_items(
        LONGFORM_RSS, longform_target, item_type="article", publication="Longform.org"
    )
    print(f"  longform: {len(longform_items)} items")
    time.sleep(1.0)

    print(f"writing: fetching New Yorker Fiction RSS (target {nyf_target})")
    nyf_items = _fetch_rss_items(
        NEW_YORKER_FICTION_RSS,
        nyf_target,
        item_type="essay",
        publication="The New Yorker",
        fallback_creator="The New Yorker",
    )
    print(f"  new yorker fiction: {len(nyf_items)} items")
    time.sleep(1.0)

    print(f"writing: scraping Poetry Foundation popular poems (target {poetry_target})")
    poetry_items = _fetch_poetry_foundation(poetry_target)
    print(f"  poetry foundation: {len(poetry_items)} items")

    plan = longform_items + poetry_items + nyf_items
    print(f"writing: total fetched = {len(plan)}")

    written = 0
    for offset, item in enumerate(plan, start=1):
        item_id = f"writing_{offset:04d}"
        if item_id in existing_ids:
            continue
        item["id"] = item_id
        item["popularity_score"] = 0.5  # spec: no objective metric, use uniform 0.5
        record = {
            "id": item["id"],
            "modality": "writing",
            "title": item["title"],
            "creator": item["creator"],
            "year": item["year"],
            "description": item["description"],
            "reviews": item["reviews"],
            "popularity_score": item["popularity_score"],
            "cover_url": item["cover_url"],
            "external_url": item["external_url"],
            "modality_specific": item["modality_specific"],
        }
        append_jsonl(out_path, record)
        written += 1

    print(f"writing: wrote {written} new items (skipped {len(plan) - written} already present)")


# ---------------------------------------------------------------------------
# Music — implemented later in Phase B
# ---------------------------------------------------------------------------


def collect_music(target_count: int) -> None:
    raise NotImplementedError("collect_music is implemented later in Phase B.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


SOURCE_DEFAULTS = {
    "books": 600,
    "films": 600,
    "music": 1500,
    "writing": 1300,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw catalog data per modality.")
    parser.add_argument(
        "--source",
        required=True,
        choices=["books", "films", "music", "writing"],
        help="Which catalog source to collect.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="How many items to collect. Defaults to the spec value for the source.",
    )
    args = parser.parse_args()

    target = args.target_count if args.target_count is not None else SOURCE_DEFAULTS[args.source]

    if args.source == "books":
        collect_books(target)
    elif args.source == "films":
        collect_films(target)
    elif args.source == "music":
        collect_music(target)
    elif args.source == "writing":
        collect_writing(target)


if __name__ == "__main__":
    main()
