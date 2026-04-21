"""Precompute LLM profile features for each paraphrase query (Phase E1).

Usage:
    python scripts/featurize_queries.py --step {profile,build,all} [--concurrency N]

Runs each paraphrase query_text through the same LLM profile pipeline as
scripts/generate_profiles.py (with a query-framed user prompt; same SYSTEM_PROMPT),
then encodes the resulting vibe_summary with sentence-transformers and writes
data/processed/paraphrase_queries_featurized.npz per spec §5.4.

Intermediate per-query profile cache is kept at paraphrase_query_profiles.jsonl
for resumability.

Modality for queries: queries are nominally modality-agnostic. build_npz()
explicitly zeros modality_onehot to avoid label leakage during inference; the
source-item modality is stored only as metadata for stratified splitting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.backend.llm_client import get_client  # noqa: E402
from generate_profiles import (  # noqa: E402
    SYSTEM_PROMPT,
    MODEL,
    VALID_TAGS,
    _strip_code_fence,
    _validate_profile,
)

PARAPHRASE_JSONL = REPO_ROOT / "data" / "processed" / "paraphrase_queries.jsonl"
PROFILES_CACHE = REPO_ROOT / "data" / "processed" / "paraphrase_query_profiles.jsonl"
FEATURES_OUT = REPO_ROOT / "data" / "processed" / "paraphrase_queries_featurized.npz"

TAG_TO_IDX = {tag: i for i, tag in enumerate(VALID_TAGS)}
MODALITY_ORDER = ["book", "film", "music", "writing"]
MODALITY_TO_IDX = {m: i for i, m in enumerate(MODALITY_ORDER)}


def build_query_user_prompt(query_text: str) -> str:
    """Frame the query as a content description so SYSTEM_PROMPT schema still applies."""
    return (
        "The following is a short description of what a user wants to experience. "
        "Produce a Content Profile capturing the mood, intent, and aesthetic the user is "
        "seeking (not what the user IS). Treat the description AS IF it were itself a "
        "content item whose felt-experience matches what the user wants.\n\n"
        f"User's description:\n{query_text}\n\n"
        "Produce the Content Profile as JSON."
    )


async def profile_query(query_text: str, query_id: str) -> "tuple[Dict[str, Any], int]":
    client = get_client()
    user_prompt = build_query_user_prompt(query_text)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            if not response.choices:
                err = getattr(response, "error", None) or {}
                raise RuntimeError(f"empty choices (likely safety block): {err}")
            raw = _strip_code_fence((response.choices[0].message.content or "").strip())
            profile = json.loads(raw)
            _validate_profile(profile, query_id)
            profile["query_id"] = query_id
            return profile, attempt
        except Exception as e:
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"profile_query failed for {query_id} after 3 attempts: {last_exc}") from last_exc


async def run_query_profile_step(concurrency: int) -> None:
    if not PARAPHRASE_JSONL.exists():
        print(
            f"ERROR: {PARAPHRASE_JSONL} not found; run generate_profiles.py --step=paraphrase first",
            file=sys.stderr,
        )
        sys.exit(1)

    queries: List[Dict[str, Any]] = []
    with PARAPHRASE_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"featurize: loaded {len(queries)} paraphrase queries")

    done_ids = set()
    if PROFILES_CACHE.exists():
        with PROFILES_CACHE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [q for q in queries if q["query_id"] not in done_ids]
    print(
        f"featurize: {len(todo)} queries to profile ({len(queries) - len(todo)} done, "
        f"concurrency={concurrency})"
    )
    if not todo:
        return

    PROFILES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    succeeded = 0
    failed = 0
    retry_histogram = [0, 0, 0]
    t0 = time.time()

    async def work(q: Dict[str, Any]) -> None:
        nonlocal succeeded, failed
        async with sem:
            try:
                profile, retries = await profile_query(q["query_text"], q["query_id"])
            except Exception as exc:
                failed += 1
                print(f"  FAILED {q['query_id']}: {exc}", file=sys.stderr, flush=True)
                return
        retry_histogram[retries] += 1
        profile["item_id"] = q["item_id"]
        profile["modality"] = q["modality"]
        async with write_lock:
            with PROFILES_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(profile, ensure_ascii=False) + "\n")
        succeeded += 1
        if succeeded % 250 == 0:
            elapsed = time.time() - t0
            print(
                f"  progress: {succeeded}/{len(todo)} "
                f"(failed={failed}, retries={retry_histogram[1] + retry_histogram[2]}, "
                f"{elapsed / succeeded:.2f}s/item)",
                flush=True,
            )

    await asyncio.gather(*(asyncio.create_task(work(q)) for q in todo))
    elapsed = time.time() - t0
    print(
        f"featurize: done. succeeded={succeeded} failed={failed} "
        f"elapsed={elapsed:.1f}s avg={elapsed / max(1, succeeded):.2f}s/item"
    )
    print(
        f"featurize: retry histogram — first-try={retry_histogram[0]} "
        f"1-retry={retry_histogram[1]} 2-retries={retry_histogram[2]}"
    )


def build_npz() -> None:
    from sentence_transformers import SentenceTransformer

    if not PROFILES_CACHE.exists():
        print(f"ERROR: {PROFILES_CACHE} not found; run --step=profile first", file=sys.stderr)
        sys.exit(1)

    profiles: List[Dict[str, Any]] = []
    with PROFILES_CACHE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))
    print(f"featurize/npz: loaded {len(profiles)} query profiles")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vibe_texts = [p["vibe_summary"] for p in profiles]
    print(f"featurize/npz: encoding {len(vibe_texts)} vibe_summaries")
    vibe_embs = model.encode(
        vibe_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    query_ids = np.array([p["query_id"] for p in profiles])
    item_ids = np.array([p["item_id"] for p in profiles])
    modalities = np.array([p["modality"] for p in profiles])
    moods = np.array([p["mood_vector"] for p in profiles], dtype=np.float32)
    intents = np.array([p["intent_vector"] for p in profiles], dtype=np.float32)

    tag_onehot = np.zeros((len(profiles), len(VALID_TAGS)), dtype=np.float32)
    rogue = 0
    for row, p in enumerate(profiles):
        for t in p.get("aesthetic_tags", []):
            idx = TAG_TO_IDX.get(t)
            if idx is None:
                rogue += 1
                continue
            tag_onehot[row, idx] = 1.0

    # Query modality_onehot is intentionally left as all-zeros. Real user queries
    # do not carry a modality label, so attributing the source-item modality to
    # the query would create a train/inference distribution mismatch. InfoNCE
    # training with per-query modality=source-item-modality discovered a trivial
    # w_mod=0.989 shortcut in the initial KNN run; zeroing modality_onehot forces
    # the KNN loss to learn from vibe/mood/intent/tag signals instead.
    # The 'modalities' string array above still records source modality for
    # stratified train/val/test split purposes; only the feature vector is zero.
    modality_onehot = np.zeros((len(profiles), len(MODALITY_ORDER)), dtype=np.float32)

    np.savez(
        FEATURES_OUT,
        query_ids=query_ids,
        item_ids=item_ids,
        modalities=modalities,
        vibe_embeddings=vibe_embs,
        mood_vectors=moods,
        intent_vectors=intents,
        tag_onehot=tag_onehot,
        modality_onehot=modality_onehot,
    )
    print(
        f"featurize/npz: wrote {FEATURES_OUT} — {len(profiles)} queries, "
        f"rogue tags dropped: {rogue}"
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Featurize paraphrase queries.")
    parser.add_argument("--step", choices=["profile", "build", "all"], default="all")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    if args.step in ("profile", "all"):
        asyncio.run(run_query_profile_step(args.concurrency))
    if args.step in ("build", "all"):
        build_npz()


if __name__ == "__main__":
    main()
