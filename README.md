# Wave

**Cross-modal mood-aligned content recommendation.** Describe how you want to feel, get a book, film, song, and piece of writing that share the same aesthetic.

## Live demo

| | URL |
|---|---|
| Frontend | <https://wave-recsys.vercel.app> |
| Backend (OpenAPI docs) | <https://wave-recsys-production.up.railway.app/docs> |
| Runtime artifacts (HF Dataset) | <https://huggingface.co/datasets/YifeiGuo/wave-artifacts> |

> First request may take 3–5 seconds as the Railway container wakes and the LLM generates the content profile. Subsequent requests hit the cache.

<!-- ![Wave results for "something quiet and slow, like rain on glass"](docs/screenshot.png) -->

## Motivation

Most consumer recommenders surface a single affect axis per interaction — Spotify's "mood" playlists, Netflix genre rows, Goodreads similar-reads. This collapses the difference between "I am sad" and "I want to be held while I am sad."

Two choices drive the system:

1. **Mood–intent dual-axis decoupling.** Every item is represented by a 12-dimensional continuous mood vector (melancholy↔joy, calm↔intense, grounded↔dreamlike, …) *and* a 7-dimensional intent vector (Heal / Escape / Focus / Energize / Reflect / Inspire / Accompany), plus a 24-tag aesthetic vocabulary and a free-text vibe summary — all produced by a single locked LLM prompt run over item metadata. The ablation study shows that removing the intent axis drops retrieval NDCG@10 by roughly one fifth, so the two axes carry independent signal.
2. **Modality-aware item representations.** The item tower concatenates a modality one-hot into the feature vector, so "calm-tender" can encode a jazz ballad, an autumn walking poem, or a slow domestic novel as different but aesthetically-aligned points in the shared embedding space. The model is trained via InfoNCE contrastive loss on LLM-paraphrased queries, with the paraphrase query side deliberately *not* given a modality — so the model must learn cross-modal affect, not a modality-to-modality lookup.

## System architecture

```
      +-------------------+       +-----------------------+
      | Raw per-modality  |       | LLM (Gemini 3.1 Flash |
      | jsonl             | ----> | Lite Preview via      |
      | books/films/music |       | OpenRouter)           |
      | writing           |       +-----------+-----------+
      +-------------------+                   |
                                              v
                                   +---------------------+
                                   | Content Profiles    |
                                   | (vibe / mood-12 /   |
                                   |  intent-7 / tags-24)|
                                   +-----------+---------+
                                               |
                              +----------------+-----------------+
                              v                                  v
                    +---------------------+        +--------------------------+
                    | features.npz        |        | paraphrase_queries.jsonl |
                    | (N_items x all axes)|        | (3 per item, LLM-gen)    |
                    +-----------+---------+        +-------------+------------+
                                |                                |
                                v                                v
                          +---------------+          +------------------------+
                          | popularity    |          | Two-Tower (InfoNCE)    |
                          | gradient KNN  |          | QueryTower + ItemTower |
                          +---------------+          +------------------------+
                                \                          /
                                 \--- 4-layer eval --------/
                                 v
                          +------------------+
                          | FastAPI backend  |  <--- Vite+React frontend (Vercel)
                          | (Railway)        |
                          +------------------+
```

### Data sources (3,347 items)

| Modality | Count | Sources |
|---|---|---|
| Book | 600 | goodbooks-10k (top-rated) + Open Library (descriptions) |
| Film / TV | 600 | TMDB `/top_rated` (400 film + 200 TV) + reviews |
| Music | 944 | Spotify track-search seeds (15 genre×year combinations) + Last.fm `track.getInfo` for tags & wiki |
| Writing | 1,203 | 7 essay/article RSS feeds (Aeon, Literary Hub, The Paris Review, and others) + Gutendex (Project Gutenberg essays) + PoetryDB (classical poems) |

## Key results

Test split: 1,006 paraphrase queries held out (80/10/10 stratified by modality). All metrics below are on this test set using the **final Two-Tower configuration (embed_dim=64, 40 epochs)** unless stated otherwise.

| Model | NDCG@10 | P@10 | L2 coherence | L3 entropy (top-20) | L4 judge (coh / align) |
|---|---:|---:|---:|---:|---:|
| popularity baseline | 0.0013 | 0.0020 | 0.074 | 0.613 | 2.14 / 1.48 |
| gradient-weighted k-NN | 0.0732 | 0.1402 | **0.146** | **0.757** | **4.54 / 4.74** |
| **Two-Tower (final)** | **0.2105** | **0.3877** | 0.133 | 0.450 | 4.32 / 4.36 |
| Two-Tower — no-intent ablation | 0.1645 | 0.3022 | 0.141 | 0.463 | — |

(L3 max entropy is `log(4) ≈ 1.386`. Judge is `openai/gpt-5.4-nano`, chosen as an independent model family to mitigate self-preference bias against the Gemini-generated profiles.)

Three take-aways that structure the report:

- **Two-Tower lifts NDCG@10 by 162× over popularity and ~2.9× over KNN.** The no-intent ablation (trained at the same `d128/e20` config as the base it is ablated against) drops NDCG@10 from 0.2002 to 0.1645 — a **~18% drop** from removing just the 7-d intent features — so mood-intent decoupling carries independent signal.
- **Smaller embeddings won.** The hyperparameter sweep over `embed_dim ∈ {64, 128, 256, 512}` at 40 epochs picked `embed_dim=64` as best; larger capacities overfit on a ~10 k-query training set. See `data/outputs/hyperparam_sweep.png`.
- **Cross-modal transfer has real cost.** Holding out one modality at training time drops NDCG@10 by 30–51% on the full test set (`data/outputs/cross_modal_transfer.json`). The model benefits meaningfully from modality-aware item features — but that benefit is partly paid for in modality-agnostic generalization, which matters for commercial scoping and is honestly flagged in Limitations.

A precision-coverage tension is visible across layers: KNN slightly edges Two-Tower on cross-modal coherence (L2) and LLM-judge scores (L4) because near-uniform weights favour thematic blending, whereas Two-Tower's InfoNCE loss optimizes precise query-to-item retrieval. Both stories are real findings documented in `data/outputs/eval_results.json`.

## Repo structure

```
wave-recsys/
├── app/
│   ├── backend/                   # FastAPI + inference engine
│   │   ├── main.py                # /health, /api/recommend, CORS, cache
│   │   ├── inference.py           # InferenceEngine + HF artifact bootstrap
│   │   ├── models.py              # Pydantic request/response schemas
│   │   └── llm_client.py          # Async OpenRouter wrapper
│   └── frontend/                  # Vite + React + Tailwind
│       └── src/                   # App.tsx, api.ts, types.ts, index.css
├── scripts/
│   ├── collect.py                 # data collection per modality
│   ├── features.py                # catalog unify + features.npz build
│   ├── generate_profiles.py       # LLM profile + paraphrase generation
│   ├── featurize_queries.py       # paraphrase query → feature vectors
│   ├── train.py                   # KNN + Two-Tower training
│   ├── evaluate.py                # 4-layer eval + case studies
│   └── experiment.py              # hyperparam sweep + cross-modal transfer
├── data/
│   ├── raw/{books,films,music,writing}/ .gitkeep + raw.jsonl (gitignored)
│   ├── processed/                 # catalog.jsonl / features.npz (gitignored)
│   └── outputs/                   # committed eval json + plots
├── models/
│   ├── knn_weights/weights.json   # committed (small text)
│   └── two_tower/*.pt             # gitignored — hosted on HF
├── Dockerfile                     # backend image for Railway
├── railway.json                   # Railway deploy config
├── requirements.txt               # Python deps
└── .env.example                   # API-key placeholder
```

Heavy runtime artifacts (`catalog.jsonl`, `features.npz`, `model.pt`, `config.pt`, `weights.json`, `profiles.jsonl`) are not checked in — they live in the HuggingFace dataset and are downloaded by the backend on startup via `InferenceEngine.download_artifacts_if_missing()`.

## Local development

**Prerequisites:** Python 3.11+, Node 18+, an OpenRouter API key. Optionally TMDB / Last.fm / Spotify keys to *re-run data collection* (not needed for serving).

### 1. Clone and set env

```bash
git clone https://github.com/SophiaYifei/wave-recsys.git
cd wave-recsys
cp .env.example .env   # fill in the keys
```

### 2. Backend

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Quickest path: let the backend pull artifacts from HF on first boot.
export HF_REPO_ID=YifeiGuo/wave-artifacts
uvicorn app.backend.main:app --reload      # http://localhost:8000
```

This downloads ~60 MB of catalog + features + model weights into `data/processed/` and `models/` on first run, then caches them locally.

To reproduce the pipeline from scratch instead of downloading:

```bash
# 1. collect raw data (requires all API keys)
python scripts/collect.py --source=books --target-count=600
python scripts/collect.py --source=films --target-count=600
python scripts/collect.py --source=music --target-count=1500
python scripts/collect.py --source=writing --target-count=1250

# 2. unify into catalog
python scripts/features.py --step=unify

# 3. LLM profile + paraphrase generation (~$4 at current OpenRouter pricing)
python scripts/generate_profiles.py --step=profile
python scripts/generate_profiles.py --step=paraphrase
python scripts/featurize_queries.py --step=all

# 4. build item features + train
python scripts/features.py --step=build
python scripts/train.py --model=knn
python scripts/train.py --model=two_tower --embed-dim 64 --max-epochs 40

# 5. full eval
python scripts/evaluate.py --model=all --judge-queries=50

# 6. optional: hyperparam sweep + cross-modal transfer
python scripts/experiment.py --type=hyperparam_sweep
python scripts/experiment.py --type=cross_modal_transfer
```

### 3. Frontend

```bash
cd app/frontend
npm install
npm run dev       # http://localhost:5173
```

The frontend talks to `http://localhost:8000` by default; point it at a different backend with `VITE_API_BASE=<url>` in the env.

## Deployment

The production deployment is split across three services:

| Component | Host | Notes |
|---|---|---|
| Backend | Railway (Docker) | `Dockerfile` + `railway.json`; on first boot runs `download_artifacts_if_missing()` |
| Frontend | Vercel | Vite auto-detected; `VITE_API_BASE` points at the Railway URL |
| Artifacts | HuggingFace Dataset | `YifeiGuo/wave-artifacts` (public) |

### Env vars

**Railway** (backend):

- `HF_REPO_ID=YifeiGuo/wave-artifacts`
- `OPENROUTER_API_KEY=sk-or-…`
- `CORS_ALLOWED_ORIGINS=https://wave-recsys.vercel.app,http://localhost:5173`
- `HF_TOKEN=` (only if the HF dataset is private; ours is public so leave unset)

**Vercel** (frontend):

- `VITE_API_BASE=https://wave-recsys-production.up.railway.app`

### Artifact layout on HF

The HF dataset stores artifacts flat at the repo root (`catalog.jsonl`, `features.npz`, `model.pt`, `config.pt`, `weights.json`, `profiles.jsonl`). The backend's `ARTIFACTS` table maps each flat filename to the canonical nested local path, so re-uploading or reorganizing the HF side doesn't require backend code changes.

## Evaluation methodology

`scripts/evaluate.py` runs four complementary layers on the test split:

1. **Retrieval metrics** — P@K, R@K, NDCG@K, MAP@K at K ∈ {5, 10}. Each paraphrase query has exactly one gold item (its source), so P@K = R@K here.
2. **Independent-encoder coherence** — pick each model's top-1 per modality, re-encode those four items' *raw* `title + description` with a frozen `all-MiniLM-L6-v2`, and report the mean pairwise cosine. The raw-text encoder has never seen the LLM-generated vibe summaries, so this is a circularity-free test of whether the cross-modal set hangs together aesthetically.
3. **Modality-coverage entropy** — Shannon entropy of the modality distribution in the top-20. `max = log(4) ≈ 1.386` when perfectly uniform across all four modalities. This surfaces modality collapse.
4. **LLM judge** — `openai/gpt-5.4-nano` scores 50 random test queries × 3 models on internal coherence and query alignment (0–10 each). GPT-5.4-nano is chosen specifically because it is a different model family from the profile-generation LLM (Gemini), which mitigates self-preference bias against Gemini-authored profiles.

`scripts/experiment.py` adds two follow-ons:

- `--type=hyperparam_sweep` trains Two-Tower at `embed_dim ∈ {64, 128, 256, 512}` × `max_epochs ∈ {20, 40, 60}`, writes results to JSON + saves the curve as `hyperparam_sweep.png`, and promotes the best config as the canonical `model.pt`.
- `--type=cross_modal_transfer` trains four Two-Towers with one modality held out of training at a time, then scores each on the full test split (including the held-out queries). The NDCG@10 gap vs the baseline quantifies how much the model relies on seeing every modality.

10 qualitative case studies are also emitted (`data/outputs/case_studies.json`), sampled from 20 probe queries across four interest categories: `modality_collapse`, `low_coherence`, `high_cross_modal_coherence`, and `knn_two_tower_disagreement`. These are the raw material for the Error Analysis section of the report.

## Known limitations

- **Spotify audio-features endpoint is blocked (403).** Documented as a platform-level constraint; music features lean on Last.fm tags + Spotify artist genres rather than programmatic audio analysis.
- **Editorial Spotify playlists blocked (Nov 2024).** The original data plan used 8 seed playlists; they return null/OAuth-only. Pivoted to `genre:X year:YYYY-YYYY` track search, which is noisier — a `genre:hip-hop` query occasionally returns Spanish-language pop.
- **Modality imbalance.** 944 music tracks (target was 1,500; a Spotify account-level rate limit was hit at 944 during collection) vs 600 books / 600 films / 1,203 writing. Metrics are reported per-model on the stratified test split rather than micro-averaged.
- **Cross-modal transfer gap of 30–51% NDCG@10.** Holding out one modality at training time hurts retrieval materially on the full test split. The Two-Tower's modality-aware item features buy retrieval precision, but zero-shot generalization to an unseen modality is not a strong suit.
- **Evaluation uses LLM-generated paraphrase queries as ground truth.** The test queries are produced by the same profile-generation LLM (Gemini 3.1 Flash Lite Preview) that encodes the item side, so the paraphrase-recovery layer (L1) captures how well the Two-Tower inverts the LLM's own embedding of the content rather than how well recommendations match a human's intuitive affect mapping. The independent-encoder coherence layer (L2), modality entropy (L3), and cross-family LLM judge (L4) are partial hedges against this circularity, but a human-annotated test set would be strictly more informative.
- **KNN converges to near-uniform weights.** At training time the 5-way simplex model wants to collapse onto whichever single similarity is most informative; with query modality correctly zeroed (to avoid a train-inference distribution mismatch), the best-validation checkpoint sits near `[0.19, 0.17, 0.18, 0.30, 0.16]`. KNN works as an honest ensemble baseline, but the result shouldn't be read as "each feature matters equally by design."
- **LLM aesthetic-tag bias.** Across 3,346 generated profiles, `tender` appears on 59% of items and `melancholic` on 54%; `cottagecore` and `japandi` appear <1% each. The 24-tag vocabulary is not uniformly exploited. Downstream similarity is not dominated by the tag channel (tag Jaccard is one of five KNN components at weight 0.30), but it is a bias worth noting.
- **1 Gemini safety refusal.** `film_0063` (Léon: The Professional) tripped `PROHIBITED_CONTENT` on profile generation. Resolved by re-running with the TMDB reviews stripped and a synthesized description, but it is a single-item workaround rather than a general fix.

## Tech stack

- **ML.** PyTorch · sentence-transformers (all-MiniLM-L6-v2) · gradient-weighted k-NN (simplex-constrained via softmax) · Two-Tower with InfoNCE.
- **LLM.** `google/gemini-3.1-flash-lite-preview` via OpenRouter for profile + paraphrase + `why_this` captions + multimodal image queries. `openai/gpt-5.4-nano` as the independent-family judge.
- **Data.** TMDB · Spotify Web API (track search + artist metadata) · Last.fm `track.getInfo` · Gutendex (Project Gutenberg) · PoetryDB · 7 essay RSS feeds · goodbooks-10k + Open Library.
- **Backend.** FastAPI · uvicorn · Pydantic · huggingface_hub (artifact bootstrap).
- **Frontend.** Vite · React 18 · Tailwind.
- **Deployment.** Railway (Dockerfile) · Vercel · HuggingFace Datasets (for the ~60 MB of runtime artifacts).

## Acknowledgements

Developed as a deep learning course final project.

Data-source credit:

- **goodbooks-10k** (Zygmunt Zając): base catalog for 600 books.
- **TMDB**: film + TV metadata, descriptions, reviews, cover art.
- **Spotify**: track metadata, album art, popularity scores.
- **Last.fm**: listener tags + wiki summaries for music.
- **PoetryDB**: 800 classical poems across 129 poets.
- **Project Gutenberg via Gutendex**: 300 classical essay collections.
- **Aeon, Literary Hub, The Paris Review, The Atlantic Ideas, Longform.org, Granta, Public Books**: modern essay RSS feeds.
- **Open Library**: book descriptions supplementing goodbooks-10k.

## License

MIT (see `LICENSE`).
