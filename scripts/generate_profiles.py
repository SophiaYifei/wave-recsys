"""LLM-based Content Profile and paraphrase query generation.

Note: spec §5.2 names this script `profile.py`, but that shadows Python's
stdlib `profile` module when Python auto-inserts scripts/ into sys.path[0]
on script invocation (causes torch._dynamo → cProfile → `import profile`
to resolve to this file). Renamed to `generate_profiles.py` for safety.

Usage:
    python scripts/generate_profiles.py --step {profile,paraphrase} [--concurrency N] [--ids FILE]

--step=profile: for each item in data/processed/catalog.jsonl, generate a
                Content Profile (reasoning + vibe_summary + mood_vector[12] +
                intent_vector[7] + aesthetic_tags[3..5]). Append-only writes
                to data/processed/profiles.jsonl, resumable by item id.

--step=paraphrase: for each profile, generate 3 paraphrase queries. Writes to
                   data/processed/paraphrase_queries.jsonl. (Implemented after
                   profile-step approval.)

--ids FILE:    optional path to a file with one item id per line; restricts
               processing to those ids (used for A/B sanity batches).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.backend.llm_client import get_client  # noqa: E402

load_dotenv()

CATALOG_PATH = REPO_ROOT / "data" / "processed" / "catalog.jsonl"
PROFILES_PATH = REPO_ROOT / "data" / "processed" / "profiles.jsonl"
PARAPHRASE_PATH = REPO_ROOT / "data" / "processed" / "paraphrase_queries.jsonl"

MODEL = "google/gemini-3.1-flash-lite-preview"

# Guardrail: music items whose description is this short are fallback-tier
# (album-line or tag-synth), so the LLM gets a "rely on cues / prefer moderate
# values" hint appended to the user prompt. See spec §5.1 music limitations.
SPARSE_METADATA_CHAR_THRESHOLD = 150
SPARSE_METADATA_HINT = (
    "METADATA NOTE: This item has sparse metadata. Rely on title, creator, "
    "album name cues, and genre conventions. If insufficient signal, prefer "
    "moderate values over extreme values; use common aesthetic tags rather "
    "than specific ones."
)

VALID_TAGS = [
    "liminal", "domestic", "nocturnal", "pastoral",
    "velvet", "paper", "glass", "water",
    "golden-hour", "moonlit", "neon", "monochrome",
    "maximalist", "minimalist", "sacred", "mundane",
    "tender", "melancholic", "playful", "austere",
    "dark-academia", "cottagecore", "retro-analog", "japandi",
]


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT (spec §5.2.1 — LOCKED, do not modify)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert cultural curator. Given metadata about a book, film, song, or essay, you produce a structured Content Profile describing the felt experience of consuming that work.

# Your process (MUST follow exactly)

You will be asked to output JSON with a specific schema. Before writing the JSON, internally reason through:
1. What is the dominant emotional register of this work?
2. What is the intended effect on the reader/viewer/listener?
3. What use-cases does this content naturally serve?
4. What aesthetic world does it inhabit?

Then output the JSON based on that reasoning.

# The 12 Mood Dimensions

Each is a continuous value in [0, 1]. The value 0.5 means genuinely balanced — use it sparingly, not as a default.

1. **melancholy ↔ joy**: 0 = heavy sadness/grief; 0.5 = mixed; 1 = pure joy/exuberance.
2. **calm ↔ intense**: 0 = serene, meditative; 1 = pulse-racing intensity.
3. **warm ↔ cold**: 0 = emotionally warm, embracing; 1 = emotionally cold/distant. NOTE: This is about emotional temperature, NOT color palette or setting.
4. **nostalgic ↔ futuristic**: 0 = looks backward, past-oriented; 1 = looks forward, futurist. Sci-fi that is emotionally nostalgic may still score 0.4-0.6, not 0.8+.
5. **solitary ↔ communal**: 0 = individual experience emphasized; 1 = collective/family/community emphasis.
6. **gentle ↔ harsh**: 0 = soft-edged, kind; 1 = cruel, unflinching.
7. **grounded ↔ dreamlike**: 0 = realistic, people can imagine themselves in the world; 1 = surreal, unreal, not intended to happen in real life. But a sci-fi that is futuristic but with grounded theoretical framework should still tend towards the grounded side, like 0.4-0.6.
8. **tender ↔ ironic**: 0 = sincere, vulnerable, earnest; 1 = ironic, detached, winking. Do NOT mistake complex emotion for irony.
9. **heavy ↔ breezy**: 0 = weighty, serious subject matter (existential stakes, mortality, trauma, high-stakes drama); 1 = light/frothy, low-stakes, easy enjoyment. NOTE: "breezy" means emotionally easy, NOT "visually bright" or "spatially open". A film about humanity's extinction scores 0.1 no matter how vast the visuals.
10. **familiar ↔ strange**: 0 = everyday, recognizable world; 1 = alien, defamiliarized.
11. **hopeful ↔ resigned**: 0 = fiercely hopeful, fighting against odds; 1 = deeply resigned, accepting fate. CAUTION: A film can contain hardship AND still be hopeful — hopeful means "believes good is possible."
12. **slow ↔ urgent**: 0 = slow pacing, patient; 1 = urgent, time-pressured.

# The 7 Intent Dimensions

Each is a continuous value in [0, 1], indicating how strongly this content serves that use-case.

- **Heal**: comforting during emotional pain; makes you feel held. High scorers: soft comforting works; NOT works that are emotionally overwhelming even if meaningful.
- **Escape**: transports you entirely away from your current reality. High scorers: immersive other-worlds, fantasy, deep fictional dives. NOT just "distracts you" — must genuinely transport.
- **Focus**: sustains concentrated work/study as companion. High scorers: ambient, non-demanding, low-narrative. Most narrative films score near 0.
- **Energize**: provides momentum, drive, motivational lift, make you want to do something or live a life that is more meaningful.
- **Reflect**: supports deep introspection and self-examination.
- **Inspire**: opens creative possibilities, expands sense of the possible.
- **Accompany**: provides gentle presence, companionship WITHOUT demanding deep engagement. Demanding/emotionally heavy works score LOW here even if warm. True companion content is low-friction, gentle, easy to be with.

# CRITICAL: Aesthetic Tags — USE ONLY THESE 24

You MUST choose 3-5 tags, and every tag MUST be copied EXACTLY, character-for-character, from this list. Spelling variations, new words, or tags outside this list are CRITICAL ERRORS. Before finalizing, read each tag back against the list to verify.

Valid tags (copy exactly):
liminal, domestic, nocturnal, pastoral, velvet, paper, glass, water, golden-hour, moonlit, neon, monochrome, maximalist, minimalist, sacred, mundane, tender, melancholic, playful, austere, dark-academia, cottagecore, retro-analog, japandi

Tag definitions:
- liminal: in-between spaces/states, suspended (4am airports, transit)
- domestic: interior home warmth, intimate daily life
- nocturnal: the pure quality of nighttime, deep and immersive
- pastoral: countryside, slow-paced natural rhythm
- velvet: soft, heavy, sensory immersion (jazz bars, thick curtains)
- paper: dry, light, archival, minimal tactile (old letters, libraries)
- glass: transparent, cool, reflective, brittle
- water: fluid, formless, subconscious-adjacent
- golden-hour: warm dusk light, nostalgic glow
- moonlit: cool silver light, solitary poetic
- neon: artificial urban light, tense or alienated
- monochrome: restricted palette, classical restraint
- maximalist: dense, information-rich, sensory abundance
- minimalist: sparse, restrained, negative space
- sacred: ritual-adjacent, transcendent
- mundane: poetry of ordinary moments (NOT the same as "everyday story" — must be specifically about ordinary-made-luminous)
- tender: soft, vulnerable, delicate
- melancholic: beautifully sad, elegiac
- playful: light, humorous, witty
- austere: cold, stern, unbeautified
- dark-academia: old-world scholarly, gothic intellectual
- cottagecore: rural nostalgia, handmade warmth
- retro-analog: film grain, old tech, analog texture
- japandi: Japanese minimalism + Nordic rationality

# Output Schema

Output valid JSON with exactly these keys:
{
  "reasoning": "<2-4 sentences of freeform analysis of the work's core mood, intent, and aesthetic. Write this FIRST to guide your vector assignments.>",
  "vibe_summary": "<50-80 word first-person description of the felt experience>",
  "mood_vector": [<12 floats in [0, 1], in the order listed above>],
  "intent_vector": [<7 floats in [0, 1], in order: Heal, Escape, Focus, Energize, Reflect, Inspire, Accompany>],
  "aesthetic_tags": [<3-5 strings, ONLY from the 24-word vocabulary above>]
}

Be decisive. Avoid defaulting to 0.5. If a work clearly leans one way, commit to that direction with values like 0.15 or 0.85."""


# ---------------------------------------------------------------------------
# Prompt builders + generation
# ---------------------------------------------------------------------------


def is_sparse_metadata(item: Dict[str, Any]) -> bool:
    """True if item is a music track with a fallback-tier short description."""
    if item.get("modality") != "music":
        return False
    desc = item.get("description") or ""
    return len(desc) < SPARSE_METADATA_CHAR_THRESHOLD


def build_user_prompt(item: Dict[str, Any]) -> str:
    reviews = item.get("reviews") or []
    reviews_block = (
        "\n".join("- " + r for r in reviews) if reviews else "- (no reviews available)"
    )
    prompt = (
        f"Content type: {item['modality']}\n"
        f"Title: {item['title']}\n"
        f"Creator: {item['creator']}\n"
        f"Year: {item['year']}\n\n"
        f"Description: {item['description']}\n\n"
        f"Reviews / excerpts:\n{reviews_block}\n\n"
        f"Produce the Content Profile as JSON."
    )
    if is_sparse_metadata(item):
        prompt += "\n\n" + SPARSE_METADATA_HINT
    return prompt


def _strip_code_fence(text: str) -> str:
    """Some Gemini variants still wrap JSON in ```json ... ``` fences despite
    response_format={"type": "json_object"}. Strip defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[: -3].rstrip()
    return text


def _validate_profile(profile: Dict[str, Any], item_id: str) -> None:
    """Validate profile schema. Mutates aesthetic_tags: invalid tags are filtered
    out (with a stderr warning carrying the item_id, for post-hoc LLM-violation
    stats), and acceptance requires >= 3 valid tags remaining. Anything short of
    that triggers the caller's retry loop."""
    if not ("mood_vector" in profile and isinstance(profile["mood_vector"], list)
            and len(profile["mood_vector"]) == 12):
        raise ValueError("mood_vector must be a list of 12 floats")
    if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in profile["mood_vector"]):
        raise ValueError("mood_vector values must be floats in [0, 1]")
    if not ("intent_vector" in profile and isinstance(profile["intent_vector"], list)
            and len(profile["intent_vector"]) == 7):
        raise ValueError("intent_vector must be a list of 7 floats")
    if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in profile["intent_vector"]):
        raise ValueError("intent_vector values must be floats in [0, 1]")

    if not ("aesthetic_tags" in profile and isinstance(profile["aesthetic_tags"], list)):
        raise ValueError("aesthetic_tags must be a list")
    raw_tags = profile["aesthetic_tags"]
    valid = [t for t in raw_tags if t in VALID_TAGS]
    invalid = [t for t in raw_tags if t not in VALID_TAGS]
    if invalid:
        print(
            f"  [WARN] {item_id}: aesthetic_tags filtered {invalid} (not in VALID_TAGS); "
            f"kept {valid}",
            file=sys.stderr,
            flush=True,
        )
    if len(valid) > 5:
        valid = valid[:5]
    if len(valid) < 3:
        raise ValueError(
            f"only {len(valid)} valid tags remain after filtering (need >= 3); raw={raw_tags}"
        )
    profile["aesthetic_tags"] = valid

    if not ("vibe_summary" in profile and isinstance(profile["vibe_summary"], str)):
        raise ValueError("vibe_summary must be a string")
    if not ("reasoning" in profile and isinstance(profile["reasoning"], str)):
        raise ValueError("reasoning must be a string")


async def generate_profile(item: Dict[str, Any]) -> "tuple[Dict[str, Any], int]":
    """Generate one Content Profile with 3-attempt retry + schema validation.

    Returns (profile, retries_used) where retries_used is 0 if the first attempt
    succeeded, 1 if one retry was needed, 2 if two retries. The caller uses this
    to aggregate retry stats across the batch.
    """
    client = get_client()
    user_prompt = build_user_prompt(item)
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
            raw = (response.choices[0].message.content or "").strip()
            raw = _strip_code_fence(raw)
            profile = json.loads(raw)
            _validate_profile(profile, item["id"])
            profile["id"] = item["id"]
            return profile, attempt
        except Exception as e:
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(
        f"generate_profile failed for {item['id']} after 3 attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _load_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_PATH.exists():
        print(
            f"ERROR: {CATALOG_PATH} not found; run features.py --step=unify first",
            file=sys.stderr,
        )
        sys.exit(1)
    items: List[Dict[str, Any]] = []
    with CATALOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_done_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


async def run_profile_step(
    concurrency: int,
    target_ids: Optional[Set[str]] = None,
) -> None:
    catalog = _load_catalog()
    if target_ids is not None:
        catalog = [it for it in catalog if it["id"] in target_ids]
    done = _load_done_ids(PROFILES_PATH)
    todo = [it for it in catalog if it["id"] not in done]
    print(
        f"profile: {len(todo)} items to profile "
        f"({len(catalog) - len(todo)} already done, concurrency={concurrency})"
    )
    if not todo:
        return

    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    succeeded = 0
    failed = 0
    retry_histogram = [0, 0, 0]  # [first-try, 1-retry, 2-retries]
    t0 = time.time()

    async def work(item: Dict[str, Any]) -> None:
        nonlocal succeeded, failed
        async with sem:
            try:
                profile, retries_used = await generate_profile(item)
            except Exception as exc:
                failed += 1
                print(f"  FAILED {item['id']}: {exc}", file=sys.stderr, flush=True)
                return
        retry_histogram[retries_used] += 1
        async with write_lock:
            with PROFILES_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(profile, ensure_ascii=False) + "\n")
        succeeded += 1
        if succeeded % 50 == 0:
            elapsed = time.time() - t0
            avg = elapsed / succeeded
            print(
                f"  progress: {succeeded}/{len(todo)} "
                f"(failed={failed}, retries={retry_histogram[1] + retry_histogram[2]}, "
                f"{avg:.2f}s/item)",
                flush=True,
            )

    await asyncio.gather(*(asyncio.create_task(work(it)) for it in todo))
    elapsed = time.time() - t0
    print(
        f"profile: done. succeeded={succeeded} failed={failed} "
        f"elapsed={elapsed:.1f}s avg={elapsed / max(1, succeeded):.2f}s/item"
    )
    print(
        f"profile: retry histogram — first-try={retry_histogram[0]} "
        f"1-retry={retry_histogram[1]} 2-retries={retry_histogram[2]}"
    )


MOOD_POLES = [
    ("melancholy", "joy"),
    ("calm", "intense"),
    ("warm", "cold"),
    ("nostalgic", "futuristic"),
    ("solitary", "communal"),
    ("gentle", "harsh"),
    ("grounded", "dreamlike"),
    ("tender", "ironic"),
    ("heavy", "breezy"),
    ("familiar", "strange"),
    ("hopeful", "resigned"),
    ("slow", "urgent"),
]
INTENT_NAMES = ["Heal", "Escape", "Focus", "Energize", "Reflect", "Inspire", "Accompany"]


def _format_top_moods(mood_vector: List[float]) -> str:
    """Pick 3 mood dims that deviate most from 0.5, name the leaning pole."""
    scored = sorted(
        range(len(mood_vector)),
        key=lambda i: abs(mood_vector[i] - 0.5),
        reverse=True,
    )[:3]
    parts = []
    for i in scored:
        lo, hi = MOOD_POLES[i]
        pole = hi if mood_vector[i] > 0.5 else lo
        parts.append(f"high {pole}")
    return ", ".join(parts)


def _format_top_intents(intent_vector: List[float]) -> str:
    """Pick 2 highest-scoring intent dims, return their names."""
    top = sorted(
        range(len(intent_vector)),
        key=lambda i: intent_vector[i],
        reverse=True,
    )[:2]
    return ", ".join(INTENT_NAMES[i] for i in top)


PARAPHRASE_PROMPT_TEMPLATE = """Given this Content Profile:

Title: {title}
Modality: {modality}
Vibe: {vibe_summary}
Top 3 moods: {top_moods}
Top 2 intents: {top_intents}

Write 3 different ways a real person might describe what they're looking for when they want "something like this". Be diverse in register — one casual, one poetic, one practical. Each 15-40 words. Never mention the title or creator. Focus on the felt-experience the user seeks.

Output as JSON: {{"queries": ["...", "...", "..."]}}"""


def build_paraphrase_prompt(profile: Dict[str, Any], catalog_item: Dict[str, Any]) -> str:
    return PARAPHRASE_PROMPT_TEMPLATE.format(
        title=catalog_item["title"],
        modality=catalog_item["modality"],
        vibe_summary=profile["vibe_summary"],
        top_moods=_format_top_moods(profile["mood_vector"]),
        top_intents=_format_top_intents(profile["intent_vector"]),
    )


async def generate_paraphrases(
    profile: Dict[str, Any], catalog_item: Dict[str, Any]
) -> "tuple[List[str], int]":
    """Generate 3 paraphrase queries with 3-attempt retry + light schema check."""
    client = get_client()
    prompt = build_paraphrase_prompt(profile, catalog_item)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = (response.choices[0].message.content or "").strip()
            raw = _strip_code_fence(raw)
            data = json.loads(raw)
            queries = data.get("queries")
            assert isinstance(queries, list) and len(queries) == 3, \
                f"expected 3 queries, got {len(queries) if isinstance(queries, list) else type(queries)}"
            for q in queries:
                assert isinstance(q, str) and q.strip(), "query must be non-empty string"
                # Word count buffer: spec asks 15-40, allow 8-60 to tolerate LLM variance.
                wc = len(q.split())
                assert 8 <= wc <= 60, f"query word-count {wc} outside 8-60 buffer: {q!r}"
            return queries, attempt
        except Exception as e:
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(
        f"generate_paraphrases failed for {profile['id']} after 3 attempts: {last_exc}"
    ) from last_exc


async def run_paraphrase_step(
    concurrency: int,
    target_ids: Optional[Set[str]] = None,
) -> None:
    if not PROFILES_PATH.exists():
        print(
            f"ERROR: {PROFILES_PATH} not found; run --step=profile first",
            file=sys.stderr,
        )
        sys.exit(1)

    profiles: Dict[str, Dict[str, Any]] = {}
    with PROFILES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                profiles[p["id"]] = p
    catalog = {c["id"]: c for c in _load_catalog()}

    existing_query_ids: Set[str] = set()
    if PARAPHRASE_PATH.exists():
        with PARAPHRASE_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_query_ids.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Resume: an item is done when all 3 query_ids are present.
    todo_ids: List[str] = []
    for item_id in profiles:
        if target_ids is not None and item_id not in target_ids:
            continue
        if item_id not in catalog:
            continue
        needed = {f"{item_id}_q{i}" for i in (1, 2, 3)}
        if not needed.issubset(existing_query_ids):
            todo_ids.append(item_id)

    print(
        f"paraphrase: {len(todo_ids)} items to process "
        f"({len(profiles) - len(todo_ids)} already done, concurrency={concurrency})"
    )
    if not todo_ids:
        return

    PARAPHRASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    succeeded = 0
    failed = 0
    retry_histogram = [0, 0, 0]
    t0 = time.time()

    async def work(item_id: str) -> None:
        nonlocal succeeded, failed
        profile = profiles[item_id]
        catalog_item = catalog[item_id]
        async with sem:
            try:
                queries, retries_used = await generate_paraphrases(profile, catalog_item)
            except Exception as exc:
                failed += 1
                print(f"  FAILED {item_id}: {exc}", file=sys.stderr, flush=True)
                return
        retry_histogram[retries_used] += 1
        async with write_lock:
            with PARAPHRASE_PATH.open("a", encoding="utf-8") as f:
                for i, qtext in enumerate(queries, start=1):
                    record = {
                        "query_id": f"{item_id}_q{i}",
                        "item_id": item_id,
                        "modality": catalog_item["modality"],
                        "query_text": qtext,
                        "source": "llm_paraphrase",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        succeeded += 1
        if succeeded % 100 == 0:
            elapsed = time.time() - t0
            avg = elapsed / succeeded
            print(
                f"  progress: {succeeded}/{len(todo_ids)} "
                f"(failed={failed}, retries={retry_histogram[1] + retry_histogram[2]}, "
                f"{avg:.2f}s/item)",
                flush=True,
            )

    await asyncio.gather(*(asyncio.create_task(work(iid)) for iid in todo_ids))
    elapsed = time.time() - t0
    print(
        f"paraphrase: done. succeeded={succeeded} failed={failed} "
        f"elapsed={elapsed:.1f}s avg={elapsed / max(1, succeeded):.2f}s/item"
    )
    print(
        f"paraphrase: retry histogram — first-try={retry_histogram[0]} "
        f"1-retry={retry_histogram[1]} 2-retries={retry_histogram[2]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM profile + paraphrase generation.")
    parser.add_argument("--step", required=True, choices=["profile", "paraphrase"])
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional file of item ids (one per line) to restrict processing.",
    )
    args = parser.parse_args()
    target_ids: Optional[Set[str]] = None
    if args.ids:
        with open(args.ids) as f:
            target_ids = {line.strip() for line in f if line.strip()}
        print(f"profile: restricting to {len(target_ids)} ids from {args.ids}")
    if args.step == "profile":
        asyncio.run(run_profile_step(args.concurrency, target_ids))
    else:
        asyncio.run(run_paraphrase_step(args.concurrency, target_ids))


if __name__ == "__main__":
    main()
