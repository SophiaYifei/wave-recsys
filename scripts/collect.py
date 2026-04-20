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
# Writing — spec §5.1 v1.2: 3 parallel paths
#   Path 1: modern essays/articles via aggregated RSS (7 publications)
#   Path 2: classic essay collections via Gutendex (Project Gutenberg)
#   Path 3: individual poems via PoetryDB
# ---------------------------------------------------------------------------

USER_AGENT = "wave-recsys/0.1 (academic project)"

WRITING_RSS_FEEDS: List[Tuple[str, str]] = [
    ("Aeon", "https://aeon.co/feed.rss"),
    ("Literary Hub", "https://lithub.com/feed/"),
    ("The Paris Review", "https://www.theparisreview.org/blog/feed/"),
    ("The Atlantic Ideas", "https://www.theatlantic.com/feed/channel/ideas/"),
    ("Longform.org", "https://longform.org/feed"),
    ("Granta", "https://granta.com/feed/"),
    ("Public Books", "https://www.publicbooks.org/feed/"),
]

GUTENDEX_URL = "https://gutendex.com/books/"
POETRYDB_BASE = "https://poetrydb.org"


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


def _fetch_one_rss(publication: str, url: str, max_items: int = 60) -> List[Dict[str, Any]]:
    """Pull items from a single modern-essay/article RSS feed."""
    feed = feedparser.parse(url, agent=USER_AGENT)
    items: List[Dict[str, Any]] = []
    for entry in feed.entries[:max_items]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        creator = (entry.get("author") or "").strip() or publication
        link = entry.get("link") or ""
        summary = _strip_html(entry.get("summary") or entry.get("description") or "")
        description = summary[:400] or f"An article by {creator}."
        items.append(
            {
                "modality": "writing",
                "title": title,
                "creator": creator,
                "year": _entry_year(entry),
                "description": description,
                "reviews": [],
                "popularity_score": 0.6,
                "cover_url": "",
                "external_url": link,
                "modality_specific": {
                    "type": "article",
                    "word_count": _word_count(summary),
                    "publication": publication,
                },
            }
        )
    return items


def _fetch_rss_essays(target: int) -> List[Dict[str, Any]]:
    """Path 1 — aggregate 7 modern-essay RSS feeds in parallel, dedup, cap at target."""
    all_items: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(WRITING_RSS_FEEDS)) as ex:
        futures = {
            ex.submit(_fetch_one_rss, pub, url): pub for pub, url in WRITING_RSS_FEEDS
        }
        for fut in as_completed(futures):
            pub = futures[fut]
            try:
                items = fut.result()
                print(f"  rss/{pub}: {len(items)} items", flush=True)
                all_items.extend(items)
            except Exception as exc:
                print(f"  rss/{pub}: FAILED — {exc}", file=sys.stderr)

    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for item in all_items:
        key = (item["title"].strip().lower(), item["creator"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:target]


def _fetch_gutendex_essays(target: int) -> List[Dict[str, Any]]:
    """Path 2 — classic essay collections from Project Gutenberg via Gutendex API."""
    items: List[Dict[str, Any]] = []
    next_url: Optional[str] = GUTENDEX_URL
    params: Optional[Dict[str, Any]] = {"topic": "essays", "languages": "en"}
    page_num = 0
    while next_url and len(items) < target:
        page_num += 1
        # Gutendex's search endpoint is slow (~30s per page cold) — 60s timeout required.
        resp = http_get_with_retry(next_url, params=params, timeout=60)
        if resp is None or resp.status_code != 200:
            status = resp.status_code if resp is not None else "timeout/connection error"
            print(
                f"  gutendex: page {page_num} failed (status={status}); stopping after {len(items)} items",
                file=sys.stderr,
                flush=True,
            )
            break
        try:
            data = resp.json()
        except ValueError:
            break
        for book in data.get("results", []):
            if len(items) >= target:
                break
            title = (book.get("title") or "").strip()
            if not title:
                continue
            authors_list = book.get("authors") or []
            author = (authors_list[0].get("name", "") if authors_list else "").strip()
            subjects = book.get("subjects") or []
            desc = f"Essays by {author or 'unknown'}."
            if subjects:
                desc = (desc + " Subjects: " + ", ".join(subjects[:3]))[:400]
            formats = book.get("formats") or {}
            cover_url = ""
            for key in ("image/jpeg", "image/png"):
                if key in formats:
                    cover_url = formats[key]
                    break
            ext_url = ""
            for key in ("text/html; charset=utf-8", "text/html"):
                if key in formats:
                    ext_url = formats[key]
                    break
            if not ext_url:
                gid = book.get("id")
                if gid:
                    ext_url = f"https://www.gutenberg.org/ebooks/{gid}"
            items.append(
                {
                    "modality": "writing",
                    "title": title,
                    "creator": author,
                    "year": 0,  # Gutendex does not expose publication year
                    "description": desc,
                    "reviews": [],
                    "popularity_score": 0.7,
                    "cover_url": cover_url,
                    "external_url": ext_url,
                    "modality_specific": {
                        "type": "essay",
                        "word_count": 0,
                        "publication": "Project Gutenberg",
                    },
                }
            )
        next_url = data.get("next")
        params = None  # 'next' URLs already include the query params
        time.sleep(0.5)
    return items


def _fetch_poetrydb_poems(target: int) -> List[Dict[str, Any]]:
    """Path 3 — individual poems from PoetryDB (~129 classical poets)."""
    resp = http_get_with_retry(f"{POETRYDB_BASE}/author")
    if resp is None or resp.status_code != 200:
        return []
    try:
        authors = resp.json().get("authors") or []
    except ValueError:
        return []
    if not authors:
        return []

    # Spread picks across authors so output is not dominated by one prolific poet.
    import random
    rnd = random.Random(42)
    rnd.shuffle(authors)

    items: List[Dict[str, Any]] = []
    for author in authors:
        if len(items) >= target:
            break
        author_path = requests.utils.quote(author)
        resp = http_get_with_retry(f"{POETRYDB_BASE}/author/{author_path}")
        if resp is None or resp.status_code != 200:
            continue
        try:
            poems = resp.json()
        except ValueError:
            continue
        if not isinstance(poems, list):
            continue
        for poem in poems:
            if len(items) >= target:
                break
            title = (poem.get("title") or "").strip()
            if not title:
                continue
            poem_author = (poem.get("author") or author).strip()
            lines = poem.get("lines") or []
            body_text = "\n".join(lines)
            description = body_text[:400].strip()
            if not description:
                continue
            slug_title = requests.utils.quote(title)
            slug_author = requests.utils.quote(poem_author)
            items.append(
                {
                    "modality": "writing",
                    "title": title,
                    "creator": poem_author,
                    "year": 0,
                    "description": description,
                    "reviews": [],
                    "popularity_score": 0.5,
                    "cover_url": "",
                    "external_url": f"{POETRYDB_BASE}/title,author/{slug_title};{slug_author}",
                    "modality_specific": {
                        "type": "poem",
                        "word_count": len(body_text.split()),
                        "publication": "PoetryDB",
                    },
                }
            )
        time.sleep(0.2)
    return items


def collect_writing(target_count: int = 1250) -> None:
    """Build data/raw/writing/raw.jsonl via 3 parallel paths: RSS / Gutendex / PoetryDB."""
    out_path = output_path("writing")
    existing_ids = load_existing_ids(out_path)
    print(f"writing: {len(existing_ids)} items already present in {out_path}")

    # Spec §5.1 targets: RSS ~150, Gutendex ~300, PoetryDB ~800.
    rss_target = 150
    gutendex_target = 300
    poetrydb_target = max(0, target_count - rss_target - gutendex_target)

    print(
        f"writing: 3 paths in parallel — RSS ({rss_target}) / Gutendex ({gutendex_target}) / "
        f"PoetryDB ({poetrydb_target})"
    )

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_rss = ex.submit(_fetch_rss_essays, rss_target)
        f_gut = ex.submit(_fetch_gutendex_essays, gutendex_target)
        f_pdb = ex.submit(_fetch_poetrydb_poems, poetrydb_target)
        rss_items = f_rss.result()
        gut_items = f_gut.result()
        pdb_items = f_pdb.result()

    print(
        f"writing: fetched RSS {len(rss_items)} / Gutendex {len(gut_items)} / "
        f"PoetryDB {len(pdb_items)}"
    )

    # Stable order (articles → essays → poems) so ID assignment is deterministic across reruns.
    combined = rss_items + gut_items + pdb_items
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for item in combined:
        title = item["title"].strip()
        if not title:
            continue
        key = (title.lower(), (item.get("creator") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    print(
        f"writing: {len(deduped)} items after dedup (removed {len(combined) - len(deduped)} duplicates)"
    )
    deduped = deduped[:target_count]

    written = 0
    for idx, item in enumerate(deduped, start=1):
        item_id = f"writing_{idx:04d}"
        if item_id in existing_ids:
            continue
        item["id"] = item_id
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

    print(f"writing: wrote {written} new items (skipped {len(deduped) - written} already present)")


# ---------------------------------------------------------------------------
# Music — Spotify track-search seeds + Last.fm tag/wiki enrichment
#   NOTE: spec v1.2 §5.1 called for 8 editorial seed playlists, but Spotify's
#   Nov 2024 API restrictions rendered those inaccessible (editorial playlists
#   return null in search; user clones return 401 under client-credentials).
#   Pivoted to genre+year track-search seeds. See collect_music() docstring
#   for the Limitations-chapter writeup.
# ---------------------------------------------------------------------------

# (genre, year_range) tuples for Spotify track search. 15 seeds × 120 tracks
# targets ~1800 candidates → dedup → 1500. The 3 extra seeds beyond the original
# 12 (latin / soundtrack / world) cushion against genre-label noise seen during
# probing (e.g. 'hip-hop' search returning Spanish-language pop).
MUSIC_GENRE_SEEDS: List[Tuple[str, str]] = [
    ("pop", "2015-2024"),
    ("pop", "1980-1989"),
    ("pop", "1970-1979"),
    ("rock", "1970-1999"),
    ("indie", "2010-2024"),
    ("hip-hop", "2010-2024"),
    ("r&b", "2000-2024"),
    ("jazz", "1940-2024"),
    ("classical", "1600-2024"),
    ("ambient", "1990-2024"),
    ("folk", "1960-2024"),
    ("electronic", "2000-2024"),
    ("latin", "2000-2024"),
    ("soundtrack", "1980-2024"),
    ("world", "1970-2024"),
]

LASTFM_API_BASE = "https://ws.audioscrobbler.com/2.0/"


def _get_spotify_client():
    """Build a Spotipy client via client-credentials flow (no user auth required)."""
    from spotipy import Spotify
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "ERROR: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env",
            file=sys.stderr,
        )
        sys.exit(1)
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return Spotify(auth_manager=auth, retries=3, requests_timeout=20)


def _spotify_search_tracks(
    sp,
    genre: str,
    year_range: str,
    target: int,
    start_offset: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Spotify track search with 'genre:<g> year:<r>' filter, paginated at 50/page.

    Returns (tracks, next_offset). next_offset lets a caller resume this seed
    deeper into results (used by collect_music's deep-fallback path when 15
    seeds at default depth underfill after dedup).
    """
    # Spotify tightened search in 2024-25: max `limit` is now 10 (docs still say 50,
    # but anything > 10 returns 400 "Invalid limit"), and `offset + limit` cannot
    # exceed 1000 — so the deepest valid call is offset=990, limit=10.
    LIMIT = 10
    MAX_OFFSET = 1000 - LIMIT  # 990
    q = f"genre:{genre} year:{year_range}"
    tracks: List[Dict[str, Any]] = []
    offset = start_offset
    while len(tracks) < target and offset <= MAX_OFFSET:
        try:
            page = sp.search(q=q, type="track", limit=LIMIT, offset=offset)
        except Exception as exc:
            print(
                f"  spotify/{genre} {year_range}: search failed at offset {offset} — {exc}",
                file=sys.stderr,
            )
            break
        items = (page.get("tracks") or {}).get("items") or []
        if not items:
            break
        for tr in items:
            if tr and tr.get("id"):
                tracks.append(tr)
                if len(tracks) >= target:
                    break
        offset += LIMIT
    return tracks, offset


def _lastfm_track_info(artist: str, track: str) -> Tuple[List[str], str]:
    """Call Last.fm track.getInfo; returns (toptags, wiki_summary)."""
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key or not artist or not track:
        return [], ""
    resp = http_get_with_retry(
        LASTFM_API_BASE,
        params={
            "method": "track.getInfo",
            "api_key": api_key,
            "artist": artist,
            "track": track,
            "format": "json",
            "autocorrect": 1,
        },
    )
    if resp is None or resp.status_code != 200:
        return [], ""
    try:
        data = resp.json()
    except ValueError:
        return [], ""
    tr = data.get("track") or {}
    tag_block = (tr.get("toptags") or {}).get("tag") or []
    tags: List[str] = []
    if isinstance(tag_block, list):
        for t in tag_block:
            name = (t.get("name") or "").strip() if isinstance(t, dict) else ""
            if name:
                tags.append(name)
    wiki_summary = ""
    wiki = tr.get("wiki")
    if isinstance(wiki, dict):
        wiki_summary = _strip_html(wiki.get("summary") or "")
    return tags, wiki_summary


def _music_description_fallback(
    wiki_summary: str, tags: List[str], creator: str, album: str
) -> str:
    """Pick the best available description for a track, tiered:
    (1) Last.fm wiki summary; (2) synthesized line from top Last.fm tags;
    (3) minimal line from creator + album; (4) minimal line from creator only.
    Guarantees a non-empty string whenever any metadata is present.
    """
    if wiki_summary:
        return wiki_summary
    if tags:
        top = tags[0]
        return f"A {top} track. Listener tags: {', '.join(tags[:5])}."
    if creator and album:
        return f"A song by {creator} from the album {album}."
    if creator:
        return f"A song by {creator}."
    return ""


def collect_music(target_count: int = 1500) -> None:
    """Build data/raw/music/raw.jsonl from Spotify track-search seeds + Last.fm enrichment.

    Spec deviation note (for Limitations chapter of report):
    spec v1.2 §5.1 originally specified 8 editorial seed playlists (All Out 2010s,
    Deep Focus, etc.) but those were blocked by Spotify's Nov 2024 API restrictions —
    all editorial playlists return null in search, and user clones require OAuth
    user auth (unavailable under client-credentials flow). Pivoted to direct track
    search with genre+year filters; coverage is broadly equivalent but genre labels
    are noisier (e.g. 'hip-hop' queries occasionally return Spanish-language pop).
    Downstream Content Profile generation mitigates this by using Last.fm tags +
    wiki rather than Spotify genre labels as the primary signal.

    Known limitation (collection-time reality): During actual collection on
    2026-04-19, the Spotify Web API client-credentials flow hit a rolling rate
    limit window (~23.8 hour cooldown) after ~944 tracks were successfully
    fetched. This is a documented post-Nov-2024 restriction on free-tier
    developer apps. Rather than waiting 18+ hours to resume, we accepted the
    partial collection. Music modality size: 944 (vs target 1500, 62.9%).
    Total library: 3347 items (vs planned 4000, 83.7%). Modality balance is
    actually improved by this (944 vs books/films 600 is less extreme than
    1500 vs 600). Recorded in report Limitations section.
    """
    out_path = output_path("music")
    existing_ids = load_existing_ids(out_path)
    print(f"music: {len(existing_ids)} items already present in {out_path}")

    sp = _get_spotify_client()

    # 15 seeds × 120 tracks = 1800 candidates; dedup trims toward target_count.
    per_seed = 120
    seed_offsets: List[int] = [0] * len(MUSIC_GENRE_SEEDS)

    pool: Dict[str, Dict[str, Any]] = {}
    for i, (genre, year_range) in enumerate(MUSIC_GENRE_SEEDS, start=1):
        tracks, next_offset = _spotify_search_tracks(
            sp, genre, year_range, per_seed, start_offset=seed_offsets[i - 1]
        )
        seed_offsets[i - 1] = next_offset
        added = 0
        for tr in tracks:
            tid = tr.get("id")
            if tid and tid not in pool:
                pool[tid] = tr
                added += 1
        print(
            f"music: seed {i}/{len(MUSIC_GENRE_SEEDS)} complete "
            f"({genre} {year_range}): +{added} unique (pool={len(pool)})",
            flush=True,
        )
        time.sleep(0.3)

    # Deep fallback: if dedup leaves us below target, go deeper on existing seeds
    # rather than adding new noisy seeds. Rotate through all 15 until pool >= target
    # or every seed has exhausted Spotify's offset=1000 cap.
    while len(pool) < target_count:
        made_progress = False
        for i, (genre, year_range) in enumerate(MUSIC_GENRE_SEEDS, start=1):
            if len(pool) >= target_count:
                break
            start = seed_offsets[i - 1]
            if start >= 1000:
                continue
            tracks, next_offset = _spotify_search_tracks(
                sp, genre, year_range, 60, start_offset=start
            )
            seed_offsets[i - 1] = next_offset
            added = 0
            for tr in tracks:
                tid = tr.get("id")
                if tid and tid not in pool:
                    pool[tid] = tr
                    added += 1
            if added > 0:
                made_progress = True
                print(
                    f"music: deep-fallback seed {i} ({genre} {year_range}) "
                    f"from offset {start}: +{added} unique (pool={len(pool)})",
                    flush=True,
                )
            time.sleep(0.3)
        if not made_progress:
            print(
                f"music: all seeds exhausted; final pool size = {len(pool)} "
                f"(target was {target_count})",
                flush=True,
            )
            break

    if not pool:
        print("music: no tracks retrieved from Spotify — aborting.", file=sys.stderr)
        return

    # Deterministic order: Spotify popularity desc, then track id.
    candidates = sorted(
        pool.values(),
        key=lambda t: (-(int(t.get("popularity") or 0)), t.get("id") or ""),
    )[:target_count]

    artist_genre_cache: Dict[str, str] = {}

    def _artist_genres(artist_id: Optional[str]) -> str:
        if not artist_id:
            return ""
        if artist_id in artist_genre_cache:
            return artist_genre_cache[artist_id]
        genres = ""
        try:
            payload = sp.artist(artist_id)
            gl = payload.get("genres") or []
            genres = ", ".join(gl)
        except Exception as exc:
            print(
                f"  [WARN] sp.artist({artist_id}) failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        artist_genre_cache[artist_id] = genres
        return genres

    lastfm_sleep = 1.0 / 3.0  # ~3 req/sec (spec 7.3 conservative cap)
    written = 0
    skipped = 0

    for idx, tr in enumerate(tqdm(candidates, desc="music"), start=1):
        item_id = f"music_{idx:04d}"
        if item_id in existing_ids:
            skipped += 1
            continue

        artists = tr.get("artists") or []
        primary_name = (artists[0].get("name") or "").strip() if artists else ""
        creator = ", ".join(a.get("name", "") for a in artists[:2] if a.get("name"))
        primary_artist_id = artists[0].get("id") if artists else None
        title = (tr.get("name") or "").strip()
        album_obj = tr.get("album") or {}
        album = album_obj.get("name") or ""
        images = album_obj.get("images") or []
        cover_url = images[0].get("url", "") if images else ""
        rel = (album_obj.get("release_date") or "").split("-")[0]
        try:
            year = int(rel) if rel else 0
        except ValueError:
            year = 0
        spotify_pop = int(tr.get("popularity") or 0)
        external_url = ((tr.get("external_urls") or {}).get("spotify")) or ""

        genre = _artist_genres(primary_artist_id)

        tags, wiki_summary = _lastfm_track_info(primary_name, title)
        time.sleep(lastfm_sleep)

        tag_review = ""
        if tags:
            tag_review = "Popular tags: " + ", ".join(tags[:6])
        reviews: List[str] = [tag_review] if tag_review else []

        record = {
            "id": item_id,
            "modality": "music",
            "title": title,
            "creator": creator,
            "year": year,
            "description": _music_description_fallback(wiki_summary, tags, creator, album),
            "reviews": reviews,
            "popularity_score": spotify_pop / 100.0,
            "cover_url": cover_url,
            "external_url": external_url,
            "modality_specific": {
                "genre": genre,
                "album": album,
                "lastfm_tags": tags,
                "spotify_popularity": spotify_pop,
            },
        }
        append_jsonl(out_path, record)
        written += 1

        if (idx % 50) == 0:
            print(
                f"  progress: idx={idx}/{len(candidates)} written={written} skipped={skipped}",
                flush=True,
            )

    print(f"music: done. wrote {written} new items, skipped {skipped} existing.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


SOURCE_DEFAULTS = {
    "books": 600,
    "films": 600,
    "music": 1500,
    "writing": 1250,
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
