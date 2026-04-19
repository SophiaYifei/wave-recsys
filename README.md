# Wave

Cross-modal mood-aligned content recommendation. Describe a feeling, get a book, film, song, and essay that share the same aesthetic.

## Live Demo

[frontend URL — TODO] | [backend API docs — TODO]

## Setup

1. Clone repo and create virtual env:

   ```
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   - OpenRouter (https://openrouter.ai)
   - TMDB (https://www.themoviedb.org/settings/api)
   - Last.fm (https://www.last.fm/api/account/create)
   - Spotify (https://developer.spotify.com/dashboard)

3. Pipeline:

   ```
   python scripts/collect.py --source=books --target-count=600
   python scripts/collect.py --source=films --target-count=600
   python scripts/collect.py --source=music --target-count=1500
   python scripts/collect.py --source=writing --target-count=1300
   python scripts/features.py --step=unify
   python scripts/profile.py --step=profile
   python scripts/profile.py --step=paraphrase
   python scripts/features.py --step=build
   python scripts/train.py --model=knn
   python scripts/train.py --model=two_tower
   python scripts/evaluate.py
   ```

4. Run backend:

   ```
   uvicorn app.backend.main:app --reload
   ```

5. Run frontend:

   ```
   cd app/frontend && npm install && npm run dev
   ```

## Architecture

Three recommendation models built on a shared Content Profile representation
(50–80 word vibe summary, 12-dim mood vector, 7-dim intent vector, 3–5
aesthetic tags from a controlled vocabulary):

- **Naive** — popularity ranking per modality.
- **Classical ML** — gradient-based weighted k-NN over the profile components.
- **Deep Learning** — Two-Tower neural network with modality-aware item features.

See the project report for full details.

## License

MIT
