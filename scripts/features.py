"""Catalog unification and feature matrix construction.

Usage:
    python scripts/features.py --step {unify,build}

--step=unify: merge the 4 raw/{modality}/raw.jsonl files into
              data/processed/catalog.jsonl. Item IDs are already assigned
              at collection time ({modality}_NNNN), and each raw file is
              already sorted deterministically (books: goodbooks ratings_count
              desc; films: TMDB popularity desc; music: spotify_popularity
              desc; writing: path order article->essay->poem with internal
              fetch order). Unify preserves those orders.

--step=build: embedding + feature vector construction. Implemented in Phase D.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"

MODALITIES = ["books", "films", "music", "writing"]


def _read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def unify() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "catalog.jsonl"

    totals = {}
    with out_path.open("w", encoding="utf-8") as out:
        for m in MODALITIES:
            src = RAW_DIR / m / "raw.jsonl"
            if not src.exists():
                print(f"ERROR: {src} not found", file=sys.stderr)
                sys.exit(1)
            items = _read_jsonl(src)
            totals[m] = len(items)
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = sum(totals.values())
    print(f"unify: wrote {total} items to {out_path}")
    for m in MODALITIES:
        print(f"  {m:>7}: {totals[m]}")


VALID_TAGS = [
    "liminal", "domestic", "nocturnal", "pastoral",
    "velvet", "paper", "glass", "water",
    "golden-hour", "moonlit", "neon", "monochrome",
    "maximalist", "minimalist", "sacred", "mundane",
    "tender", "melancholic", "playful", "austere",
    "dark-academia", "cottagecore", "retro-analog", "japandi",
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(VALID_TAGS)}
MODALITY_ORDER = ["book", "film", "music", "writing"]
MODALITY_TO_IDX = {m: i for i, m in enumerate(MODALITY_ORDER)}

CATALOG_PATH = PROC_DIR / "catalog.jsonl"
PROFILES_PATH = PROC_DIR / "profiles.jsonl"
FEATURES_PATH = PROC_DIR / "features.npz"


def build() -> None:
    """Join catalog + profiles, encode vibe_summary, emit features.npz per spec §4.4."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found; run --step=unify first", file=sys.stderr)
        sys.exit(1)
    if not PROFILES_PATH.exists():
        print(
            f"ERROR: {PROFILES_PATH} not found; run profile.py --step=profile first",
            file=sys.stderr,
        )
        sys.exit(1)

    catalog = {c["id"]: c for c in _read_jsonl(CATALOG_PATH)}
    profiles = {p["id"]: p for p in _read_jsonl(PROFILES_PATH)}

    # Preserve catalog order (deterministic) and drop items missing a profile.
    ordered_ids = [iid for iid in catalog if iid in profiles]
    dropped = len(catalog) - len(ordered_ids)
    if dropped:
        print(f"features/build: dropping {dropped} catalog items with no profile")
    if not ordered_ids:
        print("ERROR: no items have both a catalog entry and a profile.", file=sys.stderr)
        sys.exit(1)

    print(f"features/build: encoding {len(ordered_ids)} vibe_summaries with "
          "SentenceTransformer('all-MiniLM-L6-v2')")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vibe_texts = [profiles[iid]["vibe_summary"] for iid in ordered_ids]
    vibe_emb = model.encode(
        vibe_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    item_ids = np.array(ordered_ids)
    modalities = np.array([catalog[i]["modality"] for i in ordered_ids])

    moods = np.array(
        [profiles[i]["mood_vector"] for i in ordered_ids], dtype=np.float32
    )
    intents = np.array(
        [profiles[i]["intent_vector"] for i in ordered_ids], dtype=np.float32
    )

    tag_onehot = np.zeros((len(ordered_ids), len(VALID_TAGS)), dtype=np.float32)
    rogue_tags = 0
    for row, iid in enumerate(ordered_ids):
        for tag in profiles[iid].get("aesthetic_tags", []):
            idx = TAG_TO_IDX.get(tag)
            if idx is None:
                rogue_tags += 1
                print(
                    f"  [WARN] {iid}: build skipped invalid tag '{tag}' (not in VALID_TAGS)",
                    file=sys.stderr,
                )
                continue
            tag_onehot[row, idx] = 1.0

    modality_onehot = np.zeros((len(ordered_ids), len(MODALITY_ORDER)), dtype=np.float32)
    for row, iid in enumerate(ordered_ids):
        m = catalog[iid]["modality"]
        modality_onehot[row, MODALITY_TO_IDX[m]] = 1.0

    popularity_scores = np.array(
        [float(catalog[i].get("popularity_score") or 0.0) for i in ordered_ids],
        dtype=np.float32,
    )

    np.savez(
        FEATURES_PATH,
        item_ids=item_ids,
        modalities=modalities,
        vibe_embeddings=vibe_emb,
        mood_vectors=moods,
        intent_vectors=intents,
        tag_onehot=tag_onehot,
        modality_onehot=modality_onehot,
        popularity_scores=popularity_scores,
    )
    print(f"features/build: wrote {FEATURES_PATH} — {len(ordered_ids)} items, "
          f"rogue tags skipped: {rogue_tags}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog unification / feature build.")
    parser.add_argument("--step", required=True, choices=["unify", "build"])
    args = parser.parse_args()
    if args.step == "unify":
        unify()
    else:
        build()


if __name__ == "__main__":
    main()
