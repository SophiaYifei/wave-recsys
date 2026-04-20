"""Four-layer evaluation: retrieval, coherence, modality entropy, LLM judge.

Usage:
    python scripts/evaluate.py [--model {popularity,knn,two_tower,two_tower_no_intent,all}]
                               [--output PATH]
                               [--judge-queries N]

Implements spec §5.5 end-to-end:
- Layer 1: P@K, R@K, NDCG@K, MAP@K at K in {5, 10} per model on paraphrase
           test split
- Layer 2: Independent-encoder mean pairwise cosine over the top-1 item of
           each modality, using a frozen sentence-transformer on RAW
           title+description (not the LLM-generated vibe_summary) to avoid
           circular reasoning
- Layer 3: Modality coverage entropy over top-20 per query (max = log(4) ≈ 1.386)
- Layer 4: Cross-LLM judge with JUDGE_MODEL on `--judge-queries` random test
           queries × 3 models (popularity / knn / two_tower); see JUDGE_MODEL
           docstring below for why we pick an OpenAI model.

Also assembles 10 qualitative case studies from 20 preselected probe queries
(picking the most interesting failures/disagreements); output goes to
data/outputs/case_studies.json.

JUDGE_MODEL: spec §5.5 Layer 4 uses an LLM-as-judge. We deliberately pick an
independent model family (OpenAI) instead of the profile-generation model
(Google Gemini) to mitigate self-preference bias — the judge should not
preferentially reward content profiles produced by a sibling model.
`openai/gpt-5.4-nano` is a small OpenAI-family model with intelligence score
higher than Flash Lite (used for profile generation). Pricing: $0.20/M input +
$1.25/M output. 50 queries × 3 models × ~400 input + ~200 output tokens each
≈ $0.07 per run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.backend.llm_client import get_client  # noqa: E402
from train import (  # noqa: E402
    FEATURES_PATH,
    QUERY_FEATURES_PATH,
    KNN_WEIGHTS_PATH,
    TT_DIR,
    SPLIT_SEED,
    TwoTower,
    WeightedKNN,
    build_item_id_to_idx,
    build_splits,
    compute_retrieval_metrics,
    load_item_features,
    load_query_features,
)

load_dotenv()

JUDGE_MODEL = "openai/gpt-5.4-nano"
CATALOG_PATH = REPO_ROOT / "data" / "processed" / "catalog.jsonl"
PARAPHRASE_JSONL = REPO_ROOT / "data" / "processed" / "paraphrase_queries.jsonl"
OUTPUTS_DIR = REPO_ROOT / "data" / "outputs"
EVAL_RESULTS_PATH = OUTPUTS_DIR / "eval_results.json"
CASE_STUDIES_PATH = OUTPUTS_DIR / "case_studies.json"
MODALITY_ONEHOT_PATH_STUB = None  # unused; features.npz carries onehots

MODALITIES = ["book", "film", "music", "writing"]


# ---------------------------------------------------------------------------
# Scorers for each model family
# ---------------------------------------------------------------------------


def _load_knn_weights() -> np.ndarray:
    with KNN_WEIGHTS_PATH.open() as f:
        d = json.load(f)
    return np.array(
        [d["w_vibe"], d["w_mood"], d["w_intent"], d["w_tag"], d["w_modality"]],
        dtype=np.float32,
    )


def _cos_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    An = F.normalize(A, dim=-1)
    Bn = F.normalize(B, dim=-1)
    return An @ Bn.t()


def _jaccard_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    Ab = (A > 0).float()
    Bb = (B > 0).float()
    inter = Ab @ Bb.t()
    a_card = Ab.sum(dim=-1, keepdim=True)
    b_card = Bb.sum(dim=-1, keepdim=True).t()
    union = (a_card + b_card - inter).clamp(min=1e-8)
    return inter / union


def score_popularity(
    queries: Dict[str, Any], items: Dict[str, Any], q_idx: np.ndarray
) -> torch.Tensor:
    """Popularity baseline: score = popularity, ignores the query."""
    _ = queries, q_idx  # unused
    pop = items["popularity"].unsqueeze(0)  # (1, N)
    return pop.expand(len(q_idx), -1).clone()


def score_knn(
    queries: Dict[str, Any], items: Dict[str, Any], q_idx: np.ndarray
) -> torch.Tensor:
    w = torch.from_numpy(_load_knn_weights())
    q = {k: queries[k][q_idx] for k in ("vibe", "mood", "intent", "tag", "modality")}
    i = items
    s_v = _cos_matrix(q["vibe"], i["vibe"])
    s_m = _cos_matrix(q["mood"], i["mood"])
    s_i = _cos_matrix(q["intent"], i["intent"])
    s_t = _jaccard_matrix(q["tag"], i["tag"])
    s_mod = _cos_matrix(q["modality"], i["modality"])
    return (
        w[0] * s_v + w[1] * s_m + w[2] * s_i + w[3] * s_t + w[4] * s_mod
    )


def score_two_tower(
    queries: Dict[str, Any],
    items: Dict[str, Any],
    q_idx: np.ndarray,
    model_filename: str = "model.pt",
) -> torch.Tensor:
    """Load a trained Two-Tower and produce (Q, N) scores."""
    cfg_name = (
        "config.pt"
        if model_filename == "model.pt"
        else model_filename.replace(".pt", "_config.pt")
    )
    cfg = torch.load(TT_DIR / cfg_name, weights_only=False)
    model = TwoTower(
        use_intent=cfg["use_intent"],
        embed_dim=cfg["embed_dim"],
        temperature=cfg["temperature"],
    )
    state = torch.load(TT_DIR / model_filename, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    q = {k: queries[k][q_idx] for k in ("vibe", "mood", "intent", "tag", "modality")}
    with torch.no_grad():
        i_emb = model.encode_item(items)      # (N, D)
        q_emb = model.encode_query(q)         # (Q, D)
        return q_emb @ i_emb.t()


# ---------------------------------------------------------------------------
# Layer 1: Retrieval metrics
# ---------------------------------------------------------------------------


def layer1_retrieval(
    queries: Dict[str, Any],
    items: Dict[str, Any],
    test_idx: np.ndarray,
    test_gold: torch.Tensor,
    score_fn,
) -> Dict[str, float]:
    scores = score_fn(queries, items, test_idx)
    m10 = compute_retrieval_metrics(scores, test_gold, k=10)
    m5 = compute_retrieval_metrics(scores, test_gold, k=5)
    # R@K for single-gold equals P@K at the item-level threshold; rename for spec
    return {
        "P@5": m5["P@5"],
        "R@5": m5["P@5"],
        "NDCG@5": m5["NDCG@5"],
        "MAP@5": m5["MAP@5"],
        "P@10": m10["P@10"],
        "R@10": m10["P@10"],
        "NDCG@10": m10["NDCG@10"],
        "MAP@10": m10["MAP@10"],
    }


# ---------------------------------------------------------------------------
# Layer 2: Independent-encoder coherence
# ---------------------------------------------------------------------------


def _pick_top1_per_modality(
    scores_row: torch.Tensor, items: Dict[str, Any]
) -> Dict[str, int]:
    """Given (N,) scores for one query, pick the top-1 item index per modality."""
    top1: Dict[str, int] = {}
    modalities_arr = items["modalities"]
    for m in MODALITIES:
        mask = np.array([modalities_arr[i] == m for i in range(len(modalities_arr))])
        mask_t = torch.from_numpy(mask)
        masked = scores_row.clone()
        masked[~mask_t] = -1e9
        top1[m] = int(masked.argmax().item())
    return top1


def layer2_coherence(
    queries: Dict[str, Any],
    items: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    test_idx: np.ndarray,
    score_fn,
    raw_encoder,
) -> float:
    """Per-query: score → pick top-1 per modality → encode raw title+description
    with the INDEPENDENT sentence-transformer → mean pairwise cosine over the
    4 items. Return mean across all test queries."""
    scores = score_fn(queries, items, test_idx)
    Q = scores.shape[0]
    cohers: List[float] = []
    for q in range(Q):
        top1 = _pick_top1_per_modality(scores[q], items)
        raw_texts = []
        for m in MODALITIES:
            iid = items["item_ids"][top1[m]]
            cat_item = catalog[iid]
            title = cat_item.get("title") or ""
            desc = cat_item.get("description") or ""
            raw_texts.append(f"{title}. {desc}".strip())
        embs = raw_encoder.encode(
            raw_texts, convert_to_numpy=True, normalize_embeddings=True
        )
        # Mean pairwise cosine among 4
        sims = []
        for a in range(4):
            for b in range(a + 1, 4):
                sims.append(float(np.dot(embs[a], embs[b])))
        cohers.append(float(np.mean(sims)))
    return float(np.mean(cohers))


# ---------------------------------------------------------------------------
# Layer 3: Modality coverage entropy
# ---------------------------------------------------------------------------


def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            H -= p * math.log(p)
    return H


def layer3_modality_entropy(
    queries: Dict[str, Any],
    items: Dict[str, Any],
    test_idx: np.ndarray,
    score_fn,
    k: int = 20,
) -> float:
    scores = score_fn(queries, items, test_idx)
    topk = scores.argsort(dim=1, descending=True)[:, :k].numpy()
    modalities_arr = items["modalities"]
    entropies = []
    for row in topk:
        counts = {m: 0 for m in MODALITIES}
        for idx in row:
            counts[modalities_arr[idx]] += 1
        entropies.append(_entropy(counts))
    return float(np.mean(entropies))


# ---------------------------------------------------------------------------
# Layer 4: Cross-LLM judge
# ---------------------------------------------------------------------------


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether 4 pieces of content form a coherent "set" — as if curated by a single editor with a clear aesthetic vision.

Query: {query}

Set:
1. Book: {book_title} by {book_creator} — {book_desc}
2. Film: {film_title} ({film_year}) — {film_desc}
3. Music: {music_title} by {music_creator} — {music_desc}
4. Writing: {writing_title} by {writing_creator} — {writing_desc}

Score the set from 0 to 10 based on:
- Internal coherence: do these 4 items share an aesthetic/emotional world?
- Query alignment: does this set fit what the query asks for?

Output JSON: {{"coherence_score": <int 0-10>, "query_alignment_score": <int 0-10>, "reasoning": "..."}}"""


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def _format_judge_prompt(
    query_text: str,
    set_items: Dict[str, Dict[str, Any]],
) -> str:
    def g(mod: str, *keys) -> str:
        item = set_items[mod]
        parts = [str(item.get(k, "")) for k in keys]
        return " | ".join(p for p in parts if p)

    return JUDGE_PROMPT_TEMPLATE.format(
        query=query_text,
        book_title=set_items["book"].get("title", ""),
        book_creator=set_items["book"].get("creator", ""),
        book_desc=_truncate(set_items["book"].get("description", ""), 200),
        film_title=set_items["film"].get("title", ""),
        film_year=set_items["film"].get("year", ""),
        film_desc=_truncate(set_items["film"].get("description", ""), 200),
        music_title=set_items["music"].get("title", ""),
        music_creator=set_items["music"].get("creator", ""),
        music_desc=_truncate(set_items["music"].get("description", ""), 200),
        writing_title=set_items["writing"].get("title", ""),
        writing_creator=set_items["writing"].get("creator", ""),
        writing_desc=_truncate(set_items["writing"].get("description", ""), 200),
    )


async def _judge_once(
    query_text: str,
    set_items: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, str]:
    """Return (coherence, query_alignment, reasoning) scored by JUDGE_MODEL."""
    client = get_client()
    prompt = _format_judge_prompt(query_text, set_items)
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            if not resp.choices:
                raise RuntimeError(f"judge returned empty choices: {getattr(resp, 'error', None)}")
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`").strip()
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            data = json.loads(raw)
            c = int(data.get("coherence_score", 0))
            q = int(data.get("query_alignment_score", 0))
            r = str(data.get("reasoning", ""))
            return c, q, r
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"judge failed: {e}") from e
    raise RuntimeError("unreachable")


async def layer4_llm_judge(
    queries: Dict[str, Any],
    items: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    test_idx: np.ndarray,
    query_texts: Dict[str, str],
    scorers: Dict[str, Any],
    n_queries: int = 50,
    seed: int = SPLIT_SEED,
    concurrency: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Score n_queries random test queries × len(scorers) models.
    Returns per-model mean {coherence, alignment} score."""
    rng = np.random.default_rng(seed + 17)
    sampled = rng.choice(test_idx, size=min(n_queries, len(test_idx)), replace=False)

    # Pre-compute score matrices per model
    score_matrices = {
        name: fn(queries, items, sampled) for name, fn in scorers.items()
    }

    # Collect jobs: (model_name, q_index_in_sampled, query_text, set_items)
    jobs: List[Tuple[str, int, str, Dict[str, Dict[str, Any]]]] = []
    for name, scores in score_matrices.items():
        for qi_local, qi in enumerate(sampled):
            top1 = _pick_top1_per_modality(scores[qi_local], items)
            set_items = {m: catalog[items["item_ids"][top1[m]]] for m in MODALITIES}
            qid = queries["query_ids"][qi]
            qtext = query_texts[qid]
            jobs.append((name, int(qi), qtext, set_items))

    results: Dict[str, List[Tuple[int, int]]] = {name: [] for name in scorers}
    sem = asyncio.Semaphore(concurrency)

    async def work(job):
        name, qi, qtext, set_items = job
        async with sem:
            try:
                c, q, _ = await _judge_once(qtext, set_items)
            except Exception as e:
                print(f"  judge FAILED {name}/{qi}: {e}", file=sys.stderr)
                return
        results[name].append((c, q))

    t0 = time.time()
    print(f"layer4: {len(jobs)} judge calls ({n_queries} queries × {len(scorers)} models)")
    await asyncio.gather(*(asyncio.create_task(work(j)) for j in jobs))
    print(f"layer4: done in {time.time() - t0:.1f}s")

    out: Dict[str, Dict[str, float]] = {}
    for name, lst in results.items():
        if not lst:
            out[name] = {"coherence": 0.0, "alignment": 0.0, "n": 0}
            continue
        coh = float(np.mean([c for c, _ in lst]))
        ali = float(np.mean([q for _, q in lst]))
        out[name] = {"coherence": coh, "alignment": ali, "n": len(lst)}
    return out


# ---------------------------------------------------------------------------
# Qualitative case studies (spec §5.5)
# ---------------------------------------------------------------------------


PROBE_QUERIES = [
    "music for studying after a breakup",
    "something to watch at 3am when I can't sleep",
    "I just need to feel something",
    "energetic but not loud",
    "like a memory of a summer I never had",
    "remind me why life is worth it",
    "surprise me",
    "something quiet and slow, like rain on glass",
    "the feeling of leaving home for the first time",
    "a book that feels like a long conversation with a stranger on a train",
    "comfort reading for a bad Sunday afternoon",
    "something to stop me doomscrolling",
    "cold, clean, and precise — like the air before snow",
    "warmth without sentimentality",
    "something that earns its sadness",
    "work music that doesn't ask anything of me",
    "a film that makes small things feel huge",
    "the aesthetic of a late-night convenience store",
    "something that feels like forgiveness",
    "a poem I can fall asleep thinking about",
]


async def _profile_probe_query(query_text: str) -> Dict[str, Any]:
    """Run a probe query through the LLM profile pipeline to get its features."""
    from generate_profiles import (
        SYSTEM_PROMPT,
        MODEL,
        _strip_code_fence,
        _validate_profile,
    )
    from featurize_queries import build_query_user_prompt

    client = get_client()
    user_prompt = build_query_user_prompt(query_text)
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            if not resp.choices:
                raise RuntimeError(f"empty choices: {getattr(resp, 'error', None)}")
            raw = _strip_code_fence((resp.choices[0].message.content or "").strip())
            profile = json.loads(raw)
            _validate_profile(profile, f"probe:{query_text[:30]}")
            return profile
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"probe profile failed: {e}") from e
    raise RuntimeError("unreachable")


async def build_probe_query_features(
    probe_texts: List[str],
) -> Dict[str, Any]:
    """Feature-ize each probe query via LLM profile → sentence-transformer encode.
    Returns dict shaped like load_query_features() output (query modality=zeros)."""
    from sentence_transformers import SentenceTransformer
    from features import VALID_TAGS, TAG_TO_IDX

    profiles = []
    for q in probe_texts:
        p = await _profile_probe_query(q)
        profiles.append(p)

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    vibe = enc.encode(
        [p["vibe_summary"] for p in profiles],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    N = len(profiles)
    mood = np.array([p["mood_vector"] for p in profiles], dtype=np.float32)
    intent = np.array([p["intent_vector"] for p in profiles], dtype=np.float32)
    tag = np.zeros((N, len(VALID_TAGS)), dtype=np.float32)
    for r, p in enumerate(profiles):
        for t in p.get("aesthetic_tags", []):
            idx = TAG_TO_IDX.get(t)
            if idx is not None:
                tag[r, idx] = 1.0
    modality = np.zeros((N, 4), dtype=np.float32)  # probe queries have no modality

    return {
        "query_ids": np.array([f"probe_{i:02d}" for i in range(N)]),
        "item_ids": np.array(["" for _ in range(N)]),
        "modalities": np.array(["probe" for _ in range(N)]),
        "vibe": torch.from_numpy(vibe),
        "mood": torch.from_numpy(mood),
        "intent": torch.from_numpy(intent),
        "tag": torch.from_numpy(tag),
        "modality": torch.from_numpy(modality),
        "profiles": profiles,
    }


def _classify_case(
    coherence_per_model: Dict[str, float],
    entropy_per_model: Dict[str, float],
    top1_per_model: Dict[str, Dict[str, int]],
) -> str:
    """Pick a 'category' label describing why this case is interesting.

    Priority (highest first):
      1. knn_two_tower_disagreement: KNN vs Two-Tower differ on >=3/4 top-1 picks
      2. modality_collapse: some model's top-20 entropy < 0.2 (very concentrated)
      3. low_coherence: some model's mean pairwise cosine < 0.05
      4. high_cross_modal_coherence: all models > 0.2 coherence (positive case)
      5. partial_disagreement: KNN/TT differ on 1-2 modalities
      6. nominal: everything else
    """
    if "knn" in top1_per_model and "two_tower" in top1_per_model:
        disagree = sum(
            1 for m in MODALITIES
            if top1_per_model["knn"][m] != top1_per_model["two_tower"][m]
        )
    else:
        disagree = 0

    # Use the non-popularity models for coherence/entropy thresholds (popularity
    # is always ~0.07 coherence and doesn't provide diagnostic value).
    non_pop_coh = {k: v for k, v in coherence_per_model.items() if k != "popularity"}
    non_pop_ent = {k: v for k, v in entropy_per_model.items() if k != "popularity"}

    # Priority 1: modality_collapse — any trained model's top-20 is ≤3 modalities'
    # worth of entropy (log(3) ≈ 1.10; we use 0.7 to catch "heavy concentration")
    # Most extreme: entropy = 0 means ALL top-20 items from one modality.
    if non_pop_ent and min(non_pop_ent.values()) < 0.3:
        return "modality_collapse"

    # Priority 2: low_coherence — any trained model's top-1-per-modality set is
    # very weakly related (mean pairwise cosine < 0.10)
    if non_pop_coh and min(non_pop_coh.values()) < 0.10:
        return "low_coherence"

    # Priority 3: universal_high_coherence — ALL trained models produce
    # semantically tight cross-modal picks (rare positive case)
    if non_pop_coh and all(v > 0.15 for v in non_pop_coh.values()):
        return "high_cross_modal_coherence"

    # Priority 4: total disagreement — KNN and Two-Tower pick 4 different
    # items across all 4 modalities
    if disagree >= 4:
        return "knn_two_tower_disagreement"

    # Priority 5: partial disagreement
    if disagree >= 1:
        return "partial_disagreement"

    return "nominal"


async def build_case_studies(
    items: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    scorers: Dict[str, Any],
    raw_encoder,
) -> List[Dict[str, Any]]:
    """Run 20 probe queries × 3 models. Pick 10 most interesting cases."""
    print(f"case_studies: featurizing {len(PROBE_QUERIES)} probe queries via LLM")
    probes = await build_probe_query_features(PROBE_QUERIES)

    cases: List[Dict[str, Any]] = []
    n = len(PROBE_QUERIES)
    # A numpy index array selecting all probes
    probe_idx = np.arange(n)

    for qi in range(n):
        qi_arr = np.array([qi])
        row_scores: Dict[str, torch.Tensor] = {}
        top1_per_model: Dict[str, Dict[str, int]] = {}
        coherence_per_model: Dict[str, float] = {}
        entropy_per_model: Dict[str, float] = {}
        results_per_model: Dict[str, List[Dict[str, Any]]] = {}

        for name, fn in scorers.items():
            scores = fn(probes, items, qi_arr)  # (1, N)
            row_scores[name] = scores
            top1 = _pick_top1_per_modality(scores[0], items)
            top1_per_model[name] = top1
            # coherence
            raw_texts = []
            for m in MODALITIES:
                iid = items["item_ids"][top1[m]]
                ci = catalog[iid]
                raw_texts.append(f"{ci.get('title') or ''}. {ci.get('description') or ''}".strip())
            embs = raw_encoder.encode(raw_texts, convert_to_numpy=True, normalize_embeddings=True)
            sims = [float(np.dot(embs[a], embs[b])) for a in range(4) for b in range(a + 1, 4)]
            coherence_per_model[name] = float(np.mean(sims))
            # entropy of top-20
            top20 = scores[0].argsort(descending=True)[:20].numpy()
            counts = {m: 0 for m in MODALITIES}
            for idx in top20:
                counts[items["modalities"][idx]] += 1
            entropy_per_model[name] = _entropy(counts)
            # top-3 per modality for display
            per_mod_items: List[Dict[str, Any]] = []
            for m in MODALITIES:
                mask = np.array([items["modalities"][i] == m for i in range(len(items["modalities"]))])
                mask_t = torch.from_numpy(mask)
                masked = scores[0].clone()
                masked[~mask_t] = -1e9
                top3 = masked.argsort(descending=True)[:3].numpy().tolist()
                per_mod_items.extend(
                    {
                        "modality": m,
                        "rank": r,
                        "item_id": items["item_ids"][ii],
                        "title": catalog[items["item_ids"][ii]].get("title", ""),
                        "creator": catalog[items["item_ids"][ii]].get("creator", ""),
                    }
                    for r, ii in enumerate(top3, start=1)
                )
            results_per_model[name] = per_mod_items

        category = _classify_case(
            coherence_per_model, entropy_per_model, top1_per_model
        )

        cases.append(
            {
                "case_id": f"case_{qi:02d}",
                "query": PROBE_QUERIES[qi],
                "query_profile": {
                    "reasoning": probes["profiles"][qi].get("reasoning", ""),
                    "vibe_summary": probes["profiles"][qi].get("vibe_summary", ""),
                    "mood_vector": probes["profiles"][qi].get("mood_vector", []),
                    "intent_vector": probes["profiles"][qi].get("intent_vector", []),
                    "aesthetic_tags": probes["profiles"][qi].get("aesthetic_tags", []),
                },
                "coherence_scores": coherence_per_model,
                "modality_entropy": entropy_per_model,
                "results_per_model": results_per_model,
                "category": category,
            }
        )

    # Debug: print all 20 cases' category before sampling
    print(f"  all {len(cases)} probe query categories (pre-sampling):")
    from collections import Counter
    all_cats = Counter(c["category"] for c in cases)
    for cat, n in all_cats.most_common():
        print(f"    {cat}: {n}")

    # Category-aware sampling: the priority order puts "collapse" and
    # "low_coherence" before disagreement because disagreement is a weak signal
    # (KNN and Two-Tower differ on most probes by construction); the real
    # diagnostic value is in modality collapse + coherence extremes.
    priority_order = [
        "modality_collapse",
        "low_coherence",
        "high_cross_modal_coherence",
        "knn_two_tower_disagreement",
        "partial_disagreement",
        "nominal",
    ]
    by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in priority_order}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c)

    # Round-robin with escalation: first take 1 per category (priority order)
    # then 2, 3, 4, ... up to each category's supply, until we have 10 total.
    selected: List[Dict[str, Any]] = []
    round_limit = 0
    while len(selected) < 10:
        round_limit += 1
        made_progress = False
        for cat in priority_order:
            if len(selected) >= 10:
                break
            bucket = by_cat.get(cat, [])
            taken_from_cat = sum(1 for s in selected if s["category"] == cat)
            if taken_from_cat < round_limit and taken_from_cat < len(bucket):
                selected.append(bucket[taken_from_cat])
                made_progress = True
        if not made_progress:
            # All buckets exhausted or cap reached
            break
    return selected[:10]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with CATALOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _load_query_texts() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with PARAPHRASE_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["query_id"]] = rec["query_text"]
    return out


def _items_with_popularity(items: Dict[str, Any]) -> Dict[str, Any]:
    """Attach popularity_scores tensor to items dict (loaded separately from npz)."""
    f = np.load(FEATURES_PATH, allow_pickle=True)
    items["popularity"] = torch.from_numpy(f["popularity_scores"]).float()
    return items


async def run_evaluation(
    models_arg: str,
    output_path: Path,
    judge_queries: int,
    skip_layers: bool = False,
) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    items = _items_with_popularity(load_item_features())
    queries = load_query_features()
    catalog = _load_catalog()
    item_id_to_idx = build_item_id_to_idx(items)
    _, _, test_idx = build_splits(queries)
    test_gold = torch.tensor(
        [item_id_to_idx[str(queries["item_ids"][qi])] for qi in test_idx],
        dtype=torch.long,
    )
    print(f"evaluate: test set size = {len(test_idx)}")

    from sentence_transformers import SentenceTransformer
    raw_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    scorer_all = {
        "popularity": score_popularity,
        "knn": score_knn,
        "two_tower": lambda q, i, idx: score_two_tower(q, i, idx, "model.pt"),
        "two_tower_no_intent": lambda q, i, idx: score_two_tower(q, i, idx, "model_no_intent.pt"),
    }
    if models_arg == "all":
        scorers = scorer_all
    else:
        scorers = {models_arg: scorer_all[models_arg]}

    if skip_layers:
        print("evaluate: --skip-layers set, rebuilding only case studies")
        from sentence_transformers import SentenceTransformer as _ST  # noqa: F401
        cs_scorers = {k: v for k, v in scorers.items() if k != "two_tower_no_intent"}
        print()
        print("=" * 60)
        print("Qualitative case studies (20 probe queries → pick 10, rebuild)")
        print("=" * 60)
        cases = await build_case_studies(items, catalog, cs_scorers, raw_encoder)
        with CASE_STUDIES_PATH.open("w") as f:
            json.dump(cases, f, indent=2)
        print(f"evaluate: wrote {CASE_STUDIES_PATH} ({len(cases)} cases)")
        cat_counts: Dict[str, int] = {}
        for c in cases:
            cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
        print(f"  case categories: {cat_counts}")
        return

    results: Dict[str, Any] = {
        "layer1_retrieval": {},
        "layer2_coherence": {},
        "layer3_modality_entropy": {},
        "layer4_llm_judge": {},
    }

    # Layer 1
    print("=" * 60)
    print("Layer 1: Retrieval metrics")
    print("=" * 60)
    for name, fn in scorers.items():
        m = layer1_retrieval(queries, items, test_idx, test_gold, fn)
        results["layer1_retrieval"][name] = m
        print(f"  {name:<22s} NDCG@10={m['NDCG@10']:.4f} P@10={m['P@10']:.4f} "
              f"MAP@10={m['MAP@10']:.4f} NDCG@5={m['NDCG@5']:.4f}")

    # Layer 2
    print()
    print("=" * 60)
    print("Layer 2: Independent-encoder coherence (frozen ST on RAW text)")
    print("=" * 60)
    for name, fn in scorers.items():
        c = layer2_coherence(queries, items, catalog, test_idx, fn, raw_encoder)
        results["layer2_coherence"][name] = c
        print(f"  {name:<22s} mean_pairwise_cosine={c:.4f}")

    # Layer 3
    print()
    print("=" * 60)
    print("Layer 3: Modality coverage entropy (top-20)")
    print("=" * 60)
    print(f"  (max entropy = log(4) = {math.log(4):.4f})")
    for name, fn in scorers.items():
        e = layer3_modality_entropy(queries, items, test_idx, fn, k=20)
        results["layer3_modality_entropy"][name] = e
        print(f"  {name:<22s} mean_entropy={e:.4f}")

    # Layer 4
    query_texts = _load_query_texts()
    print()
    print("=" * 60)
    print(f"Layer 4: LLM judge ({JUDGE_MODEL})")
    print("=" * 60)
    # Exclude two_tower_no_intent from judge by default to save cost (judge on 3 models per spec)
    judge_scorers = {k: v for k, v in scorers.items() if k != "two_tower_no_intent"}
    l4 = await layer4_llm_judge(
        queries, items, catalog, test_idx, query_texts,
        judge_scorers, n_queries=judge_queries,
    )
    for name, d in l4.items():
        results["layer4_llm_judge"][name] = d
        print(f"  {name:<22s} coherence={d['coherence']:.2f} "
              f"alignment={d['alignment']:.2f} (n={d['n']})")

    # Persist eval_results.json
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nevaluate: wrote {output_path}")

    # Case studies
    print()
    print("=" * 60)
    print("Qualitative case studies (20 probe queries → pick 10)")
    print("=" * 60)
    # For case studies, exclude no_intent ablation to keep output small + focused
    cs_scorers = {k: v for k, v in scorers.items() if k != "two_tower_no_intent"}
    cases = await build_case_studies(items, catalog, cs_scorers, raw_encoder)
    with CASE_STUDIES_PATH.open("w") as f:
        json.dump(cases, f, indent=2)
    print(f"evaluate: wrote {CASE_STUDIES_PATH} ({len(cases)} cases)")
    cat_counts: Dict[str, int] = {}
    for c in cases:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print(f"  case categories: {cat_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-layer evaluation.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["popularity", "knn", "two_tower", "two_tower_no_intent", "all"],
    )
    parser.add_argument("--output", type=str, default=str(EVAL_RESULTS_PATH))
    parser.add_argument("--judge-queries", type=int, default=50)
    parser.add_argument(
        "--skip-layers",
        action="store_true",
        help="Skip Layer 1-4 and only (re)build the case_studies.json.",
    )
    args = parser.parse_args()
    torch.manual_seed(SPLIT_SEED)
    asyncio.run(
        run_evaluation(args.model, Path(args.output), args.judge_queries, args.skip_layers)
    )


if __name__ == "__main__":
    main()
