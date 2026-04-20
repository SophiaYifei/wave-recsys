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
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.backend.inference import MODALITIES, get_engine  # noqa: E402
from app.backend.llm_client import get_client  # noqa: E402
from app.backend.models import (  # noqa: E402
    ProductCard,
    QueryProfile,
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
    _ = get_engine()  # eager load on startup
    print(f"[lifespan] engine ready ({time.time() - t0:.1f}s)", flush=True)
    yield


app = FastAPI(
    title="Wave Recommendations",
    version="0.1.0-mvp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


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

    try:
        profile = await _profile_user_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query profile failed: {e}")

    q_feats = engine.query_features_from_profile(profile)
    scores = engine.score(req.model, q_feats)
    top1 = engine.top_k_per_modality(scores, requested, k=1)

    # Concurrently fetch why_this for each top-1 item
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
        why_coros.append(_why_this(req.query, cat_item))

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
        )
        results[pick["m"]].append(card)

    return RecommendResponse(
        query_profile=QueryProfile(
            vibe_summary=str(profile.get("vibe_summary") or ""),
            mood_vector=[float(x) for x in profile.get("mood_vector") or []],
            intent_vector=[float(x) for x in profile.get("intent_vector") or []],
            aesthetic_tags=[str(t) for t in profile.get("aesthetic_tags") or []],
        ),
        results=results,
    )
