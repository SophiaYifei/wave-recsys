"""FastAPI entrypoint for the Wave recommendation backend (MVP).

Run locally:
    uvicorn app.backend.main:app --reload

Endpoints (MVP / Phase H tier 1):
    GET  /health            — returns {status: "ok"}
    POST /api/recommend     — run full recommend flow and return 4 cards (1 per modality)

Other spec §5.7 endpoints (/api/swap, /api/vibe-card) are not implemented in
this tier — they come after the MVP demo is approved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.backend.inference import (  # noqa: E402
    MODALITIES,
    download_artifacts_if_missing,
    get_engine,
)
from app.backend.llm_client import get_client  # noqa: E402
from app.backend.models import (  # noqa: E402
    ProductCard,
    QueryProfile,
    RecommendAllRequest,
    RecommendAllResponse,
    RecommendRequest,
    RecommendResponse,
)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def _profile_user_query(query_text: str) -> Dict[str, Any]:
    """Reuse featurize_queries.profile_query to generate a Content Profile for
    a real user query (query_id is synthetic). Errors bubble up as HTTP 500."""
    from featurize_queries import profile_query as featurize_profile
    profile, _retries = await featurize_profile(query_text, "live_query")
    return profile


async def _profile_user_query_multimodal(
    query_text: str,
    image_data_url: str,
) -> Dict[str, Any]:
    """Multimodal profile: same SYSTEM_PROMPT, but the user message contains
    an image part in addition to (optional) text. Gemini 3.1 Flash Lite Preview
    handles vision natively."""
    from generate_profiles import (
        SYSTEM_PROMPT,
        MODEL,
        _strip_code_fence,
        _validate_profile,
    )
    from featurize_queries import build_query_user_prompt

    client = get_client()
    text_for_prompt = query_text.strip() or (
        "(The user uploaded an image with no text. Interpret the mood and "
        "aesthetic conveyed by the image, and produce a Content Profile for "
        "the kind of content that would match that vibe.)"
    )
    prompt_text = build_query_user_prompt(text_for_prompt)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
            )
            if not resp.choices:
                err = getattr(resp, "error", None) or {}
                raise RuntimeError(f"multimodal profile: empty choices: {err}")
            raw = _strip_code_fence((resp.choices[0].message.content or "").strip())
            profile = json.loads(raw)
            _validate_profile(profile, "live_query_mm")
            return profile
        except Exception as e:
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(
        f"multimodal profile failed after 3 attempts: {last_exc}"
    ) from last_exc


WHY_THIS_PROMPT = """A user searched for: "{query}"

We recommended this {modality}:
  Title: {title}
  Creator: {creator}
  Year: {year}
  Description: {description}

Write ONE sentence (15-30 words, first-person "I" or second-person "you") explaining why this item fits the user's vibe. Be specific — reference the item's feel, not generic marketing copy. Do NOT mention the title or creator again.

Output JSON: {{"why_this": "<one sentence>"}}"""


async def _why_this(query: str, item: Dict[str, Any]) -> str:
    """One short caption per item. Falls back to a deterministic line on LLM failure."""
    from generate_profiles import MODEL, _strip_code_fence

    fallback = (
        f"A {item.get('modality', 'piece')} that sits near your query's "
        f"aesthetic in the model's embedding space."
    )
    client = get_client()
    prompt = WHY_THIS_PROMPT.format(
        query=query,
        modality=item.get("modality", ""),
        title=item.get("title", ""),
        creator=item.get("creator", ""),
        year=item.get("year", ""),
        description=(item.get("description") or "")[:300],
    )
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        if not resp.choices:
            return fallback
        raw = _strip_code_fence((resp.choices[0].message.content or "").strip())
        data = json.loads(raw)
        text = str(data.get("why_this", "") or "").strip()
        return text or fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# App lifespan + CORS
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    # Bootstrap: pull any missing runtime artifacts (catalog / features / model
    # weights) from HF before loading the engine. No-op when the dev machine
    # already has them on disk from local training.
    download_artifacts_if_missing()
    _ = get_engine()  # eager load on startup
    _load_cache_from_disk()
    print(f"[lifespan] engine ready ({time.time() - t0:.1f}s)", flush=True)
    yield


app = FastAPI(
    title="Wave Recommendations",
    version="0.1.0-mvp",
    lifespan=lifespan,
)

_cors_env = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
)
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# Disk-backed response cache. Key = (model, sorted-modalities, query).
# LLM calls (query profile + why_this) are non-deterministic even on identical
# input (default Gemini temperature > 0), so caching pins demo queries to
# specific saved outputs. The cache is loaded from disk at server startup and
# re-serialized to disk after every miss-filled entry, so it survives uvicorn
# restarts / machine reboots. Bypass with `bypass_cache: true` in the request.
_recommend_cache: Dict[str, RecommendResponse] = {}
_cache_lock = asyncio.Lock()
RECOMMEND_CACHE_PATH = REPO_ROOT / "data" / "processed" / "recommend_cache.json"


def _cache_key(
    query: str,
    model: str,
    modalities: List[str],
    image_b64: Optional[str] = None,
) -> str:
    """Key layout: {model}|{sorted-modalities}|{imghash-or-noimg}|{query}.

    The image hash is the first 16 chars of sha256 over the full data URL;
    identical base64 (same file re-uploaded) collapses to the same cache entry.
    """
    if image_b64:
        img_hash = hashlib.sha256(image_b64.encode("utf-8")).hexdigest()[:16]
    else:
        img_hash = "noimg"
    return f"{model}|{','.join(sorted(modalities))}|{img_hash}|{query}"


def _load_cache_from_disk() -> None:
    if not RECOMMEND_CACHE_PATH.exists():
        print(f"[cache] no persisted cache at {RECOMMEND_CACHE_PATH}", flush=True)
        return
    try:
        with RECOMMEND_CACHE_PATH.open() as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[cache] load failed ({e}); starting empty", file=sys.stderr, flush=True)
        return

    engine = get_engine()  # for catalog lookups during schema migration
    loaded = 0
    migrated_cards = 0
    migrated_keys = 0
    for key, body in raw.items():
        try:
            resp = RecommendResponse.model_validate(body)
        except Exception as e:
            print(f"[cache] skipping corrupt entry {key!r}: {e}", file=sys.stderr)
            continue
        # Back-fill subtype + excerpt for cards cached before the schema added
        # those fields. Look them up from the catalog, which is stable.
        for cards in resp.results.values():
            for c in cards:
                if (not c.subtype and not c.excerpt) and c.id in engine.catalog:
                    cat = engine.catalog[c.id]
                    c.subtype = str((cat.get("modality_specific") or {}).get("type") or "")
                    c.excerpt = str(cat.get("description") or "")
                    if c.subtype or c.excerpt:
                        migrated_cards += 1
        # Key migration: legacy keys were "model|mods|query"; new keys include
        # an image-hash segment "model|mods|imghash|query". Inject "noimg".
        parts = key.split("|", 3)
        if len(parts) == 3:
            key = f"{parts[0]}|{parts[1]}|noimg|{parts[2]}"
            migrated_keys += 1
        _recommend_cache[key] = resp
        loaded += 1
    print(
        f"[cache] loaded {loaded} entries from {RECOMMEND_CACHE_PATH.name} "
        f"({migrated_cards} cards back-filled, {migrated_keys} keys migrated to imghash format)",
        flush=True,
    )


async def _save_cache_to_disk() -> None:
    """Atomic write: serialize full cache to tmp file, then rename over target."""
    async with _cache_lock:
        snapshot = list(_recommend_cache.items())

    RECOMMEND_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v.model_dump() for k, v in snapshot}
    tmp = RECOMMEND_CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
    tmp.replace(RECOMMEND_CACHE_PATH)
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "cached_queries": str(len(_recommend_cache))}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    engine = get_engine()

    requested = req.modalities or MODALITIES
    for m in requested:
        if m not in MODALITIES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown modality {m!r}; valid: {MODALITIES}",
            )

    if req.model not in {"popularity", "knn", "two_tower"}:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {req.model!r}; valid: popularity / knn / two_tower",
        )

    if not req.query.strip() and not req.image_base64:
        raise HTTPException(
            status_code=400,
            detail="at least one of `query` or `image_base64` must be non-empty",
        )

    cache_key = _cache_key(req.query, req.model, requested, req.image_base64)
    if not req.bypass_cache and cache_key in _recommend_cache:
        return _recommend_cache[cache_key]

    try:
        if req.image_base64:
            profile = await _profile_user_query_multimodal(req.query, req.image_base64)
        else:
            profile = await _profile_user_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query profile failed: {e}")

    q_feats = engine.query_features_from_profile(profile)
    scores = engine.score(req.model, q_feats)
    top1 = engine.top_k_per_modality(scores, requested, k=1)

    # Concurrently fetch why_this for each top-1 item. For image-only queries
    # we fall back to the LLM-derived vibe_summary so the why_this prompt has
    # something meaningful to reference.
    q_for_caption = req.query.strip() or str(profile.get("vibe_summary") or "")
    why_coros = []
    picks: List[Dict[str, Any]] = []
    for m in requested:
        if not top1.get(m):
            continue
        idx = top1[m][0]
        iid = str(engine.item_ids[idx])
        cat_item = engine.catalog[iid]
        sim = float(scores[idx].item())
        picks.append({"m": m, "idx": idx, "iid": iid, "cat": cat_item, "sim": sim})
        why_coros.append(_why_this(q_for_caption, cat_item))

    why_texts = await asyncio.gather(*why_coros, return_exceptions=False)

    results: Dict[str, List[ProductCard]] = {m: [] for m in requested}
    for pick, why in zip(picks, why_texts):
        cat_item = pick["cat"]
        try:
            year = int(cat_item.get("year") or 0)
        except (TypeError, ValueError):
            year = 0
        card = ProductCard(
            id=pick["iid"],
            modality=cat_item.get("modality", pick["m"]),
            title=str(cat_item.get("title") or ""),
            creator=str(cat_item.get("creator") or ""),
            year=year,
            cover_url=str(cat_item.get("cover_url") or ""),
            external_url=str(cat_item.get("external_url") or ""),
            similarity=pick["sim"],
            why_this=why,
            subtype=str((cat_item.get("modality_specific") or {}).get("type") or ""),
            excerpt=str(cat_item.get("description") or ""),
        )
        results[pick["m"]].append(card)

    response = RecommendResponse(
        query_profile=QueryProfile(
            vibe_summary=str(profile.get("vibe_summary") or ""),
            mood_vector=[float(x) for x in profile.get("mood_vector") or []],
            intent_vector=[float(x) for x in profile.get("intent_vector") or []],
            aesthetic_tags=[str(t) for t in profile.get("aesthetic_tags") or []],
        ),
        results=results,
    )
    _recommend_cache[cache_key] = response
    await _save_cache_to_disk()
    return response


@app.post("/api/recommend_all", response_model=RecommendAllResponse)
async def recommend_all(req: RecommendAllRequest) -> RecommendAllResponse:
    """Run all 3 models (popularity / knn / two_tower) in a single request.

    Saves cost vs calling /api/recommend three times: the query profile LLM
    call happens once, and why_this captions are deduplicated across models
    (if two models both pick item X as the book top-1, we only generate the
    caption once). UI uses this to let the user toggle models for free after
    a single submit.
    """
    engine = get_engine()

    requested = req.modalities or list(MODALITIES)
    for m in requested:
        if m not in MODALITIES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown modality {m!r}; valid: {MODALITIES}",
            )

    try:
        profile = await _profile_user_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query profile failed: {e}")

    q_feats = engine.query_features_from_profile(profile)

    model_names = ["two_tower", "knn", "popularity"]
    scores_by_model: Dict[str, Any] = {}
    top1_by_model: Dict[str, Dict[str, int]] = {}
    for name in model_names:
        scores = engine.score(name, q_feats)
        top1 = engine.top_k_per_modality(scores, requested, k=1)
        scores_by_model[name] = scores
        top1_by_model[name] = {m: idxs[0] for m, idxs in top1.items() if idxs}

    # Deduplicate item indices across the 3 models × requested modalities
    unique_idxs: List[int] = []
    seen: set = set()
    for per_model in top1_by_model.values():
        for idx in per_model.values():
            if idx not in seen:
                seen.add(idx)
                unique_idxs.append(int(idx))

    why_coros = []
    for idx in unique_idxs:
        iid = str(engine.item_ids[idx])
        why_coros.append(_why_this(req.query, engine.catalog[iid]))
    why_texts = await asyncio.gather(*why_coros, return_exceptions=False)
    why_by_idx: Dict[int, str] = dict(zip(unique_idxs, why_texts))

    results_by_model: Dict[str, Dict[str, List[ProductCard]]] = {}
    for name in model_names:
        bucket: Dict[str, List[ProductCard]] = {m: [] for m in requested}
        for m, idx in top1_by_model[name].items():
            iid = str(engine.item_ids[idx])
            cat_item = engine.catalog[iid]
            try:
                year = int(cat_item.get("year") or 0)
            except (TypeError, ValueError):
                year = 0
            card = ProductCard(
                id=iid,
                modality=cat_item.get("modality", m),
                title=str(cat_item.get("title") or ""),
                creator=str(cat_item.get("creator") or ""),
                year=year,
                cover_url=str(cat_item.get("cover_url") or ""),
                external_url=str(cat_item.get("external_url") or ""),
                similarity=float(scores_by_model[name][idx].item()),
                why_this=why_by_idx[idx],
                subtype=str((cat_item.get("modality_specific") or {}).get("type") or ""),
                excerpt=str(cat_item.get("description") or ""),
            )
            bucket[m].append(card)
        results_by_model[name] = bucket

    return RecommendAllResponse(
        query_profile=QueryProfile(
            vibe_summary=str(profile.get("vibe_summary") or ""),
            mood_vector=[float(x) for x in profile.get("mood_vector") or []],
            intent_vector=[float(x) for x in profile.get("intent_vector") or []],
            aesthetic_tags=[str(t) for t in profile.get("aesthetic_tags") or []],
        ),
        results_by_model=results_by_model,
    )
