"""Model loading + in-memory scoring for the recommendation backend.

Loads catalog / item features / sentence-transformer / trained Two-Tower (final
d64/e40 at models/two_tower/model.pt) at construction time. Precomputes all
item embeddings once so each request only does a single query encode + dot
product. Also supports popularity and KNN baselines for A/B comparison.

The `scripts/` directory is added to sys.path so we can reuse TwoTower and
WeightedKNN without duplicating class definitions. This is a pragmatic choice
in lieu of a shared wave/models package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train import TwoTower  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "processed" / "catalog.jsonl"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.npz"
TT_DIR = REPO_ROOT / "models" / "two_tower"
KNN_WEIGHTS_PATH = REPO_ROOT / "models" / "knn_weights" / "weights.json"

MODALITIES = ["book", "film", "music", "writing"]
MODALITY_TO_IDX = {m: i for i, m in enumerate(MODALITIES)}
VALID_TAGS = [
    "liminal", "domestic", "nocturnal", "pastoral",
    "velvet", "paper", "glass", "water",
    "golden-hour", "moonlit", "neon", "monochrome",
    "maximalist", "minimalist", "sacred", "mundane",
    "tender", "melancholic", "playful", "austere",
    "dark-academia", "cottagecore", "retro-analog", "japandi",
]
TAG_TO_IDX = {t: i for i, t in enumerate(VALID_TAGS)}


class InferenceEngine:
    def __init__(self) -> None:
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.item_ids: np.ndarray = np.array([])
        self.item_modalities: np.ndarray = np.array([])
        self.popularity: torch.Tensor = torch.tensor([])
        # raw item features (stacked tensors, aligned with item_ids)
        self.item_vibe: torch.Tensor = torch.tensor([])
        self.item_mood: torch.Tensor = torch.tensor([])
        self.item_intent: torch.Tensor = torch.tensor([])
        self.item_tag: torch.Tensor = torch.tensor([])
        self.item_modality_oh: torch.Tensor = torch.tensor([])
        # Two-Tower state
        self.tt_model: Optional[TwoTower] = None
        self.tt_config: Dict[str, Any] = {}
        self.item_embeddings: Optional[torch.Tensor] = None  # (N, D), precomputed
        # KNN weights
        self.knn_weights: Optional[np.ndarray] = None
        # Shared sentence-transformer for live query encoding
        self.encoder: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        print("[inference] loading catalog...", flush=True)
        with CATALOG_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.catalog[rec["id"]] = rec

        print("[inference] loading features.npz...", flush=True)
        npz = np.load(FEATURES_PATH, allow_pickle=True)
        self.item_ids = np.array([str(x) for x in npz["item_ids"]])
        self.item_modalities = np.array([str(x) for x in npz["modalities"]])
        self.popularity = torch.from_numpy(npz["popularity_scores"]).float()
        self.item_vibe = torch.from_numpy(npz["vibe_embeddings"]).float()
        self.item_mood = torch.from_numpy(npz["mood_vectors"]).float()
        self.item_intent = torch.from_numpy(npz["intent_vectors"]).float()
        self.item_tag = torch.from_numpy(npz["tag_onehot"]).float()
        self.item_modality_oh = torch.from_numpy(npz["modality_onehot"]).float()

        print("[inference] loading Two-Tower model.pt + config.pt...", flush=True)
        cfg = torch.load(TT_DIR / "config.pt", weights_only=False)
        self.tt_config = cfg
        model = TwoTower(
            use_intent=cfg["use_intent"],
            embed_dim=cfg["embed_dim"],
            temperature=cfg["temperature"],
        )
        state = torch.load(TT_DIR / "model.pt", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.tt_model = model

        print("[inference] precomputing item embeddings...", flush=True)
        items = {
            "vibe": self.item_vibe,
            "mood": self.item_mood,
            "intent": self.item_intent,
            "tag": self.item_tag,
            "modality": self.item_modality_oh,
        }
        with torch.no_grad():
            self.item_embeddings = model.encode_item(items)  # (N, D)

        print("[inference] loading KNN weights...", flush=True)
        with KNN_WEIGHTS_PATH.open() as f:
            kd = json.load(f)
        self.knn_weights = np.array(
            [kd["w_vibe"], kd["w_mood"], kd["w_intent"], kd["w_tag"], kd["w_modality"]],
            dtype=np.float32,
        )

        print("[inference] loading sentence-transformer (all-MiniLM-L6-v2)...", flush=True)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        print(
            f"[inference] ready: {len(self.catalog)} catalog items, "
            f"{self.item_embeddings.shape[0]} item embeddings @ dim {self.item_embeddings.shape[1]}",
            flush=True,
        )

    # ------------------------------------------------------------------ query features

    def query_features_from_profile(
        self, profile: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Turn an LLM-generated query profile into a feature dict (batch of 1)."""
        assert self.encoder is not None
        vibe = self.encoder.encode(
            profile["vibe_summary"],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        mood = np.array(profile["mood_vector"], dtype=np.float32)
        intent = np.array(profile["intent_vector"], dtype=np.float32)
        tag = np.zeros(len(VALID_TAGS), dtype=np.float32)
        for t in profile.get("aesthetic_tags", []) or []:
            idx = TAG_TO_IDX.get(t)
            if idx is not None:
                tag[idx] = 1.0
        # Real user queries have no modality — keep all-zeros (matches training featurization).
        modality = np.zeros(len(MODALITIES), dtype=np.float32)
        return {
            "vibe": torch.from_numpy(vibe).unsqueeze(0),
            "mood": torch.from_numpy(mood).unsqueeze(0),
            "intent": torch.from_numpy(intent).unsqueeze(0),
            "tag": torch.from_numpy(tag).unsqueeze(0),
            "modality": torch.from_numpy(modality).unsqueeze(0),
        }

    # ------------------------------------------------------------------ scoring

    def score_two_tower(self, q: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.tt_model is not None and self.item_embeddings is not None
        with torch.no_grad():
            q_emb = self.tt_model.encode_query(q)  # (1, D)
            return (q_emb @ self.item_embeddings.t()).squeeze(0)  # (N,)

    def score_knn(self, q: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert self.knn_weights is not None
        w = torch.from_numpy(self.knn_weights)
        qv = F.normalize(q["vibe"], dim=-1)
        iv = F.normalize(self.item_vibe, dim=-1)
        s_v = (qv @ iv.t()).squeeze(0)
        qm = F.normalize(q["mood"], dim=-1)
        im = F.normalize(self.item_mood, dim=-1)
        s_m = (qm @ im.t()).squeeze(0)
        qi = F.normalize(q["intent"], dim=-1)
        ii_ = F.normalize(self.item_intent, dim=-1)
        s_i = (qi @ ii_.t()).squeeze(0)
        # Jaccard on tag
        qt = (q["tag"] > 0).float()
        it = (self.item_tag > 0).float()
        inter = (qt @ it.t()).squeeze(0)
        q_card = qt.sum(dim=-1)
        i_card = it.sum(dim=-1)
        union = (q_card + i_card - inter).clamp(min=1e-8)
        s_t = inter / union
        qmod = F.normalize(q["modality"], dim=-1)
        imod = F.normalize(self.item_modality_oh, dim=-1)
        s_mod = (qmod @ imod.t()).squeeze(0)
        return (
            w[0] * s_v + w[1] * s_m + w[2] * s_i + w[3] * s_t + w[4] * s_mod
        )

    def score_popularity(self, q: Dict[str, torch.Tensor]) -> torch.Tensor:
        _ = q  # unused — popularity ignores the query
        return self.popularity.clone()

    def score(self, model_name: str, q: Dict[str, torch.Tensor]) -> torch.Tensor:
        if model_name == "two_tower":
            return self.score_two_tower(q)
        if model_name == "knn":
            return self.score_knn(q)
        if model_name == "popularity":
            return self.score_popularity(q)
        raise ValueError(f"unknown model: {model_name!r}")

    # ------------------------------------------------------------------ top-k

    def top_k_per_modality(
        self,
        scores: torch.Tensor,
        modalities: List[str],
        k: int = 1,
    ) -> Dict[str, List[int]]:
        """Return k item indices (highest-scoring first) per requested modality."""
        out: Dict[str, List[int]] = {}
        for m in modalities:
            mask = torch.from_numpy(self.item_modalities == m)
            if not mask.any():
                out[m] = []
                continue
            masked = scores.clone()
            masked[~mask] = -1e9
            topk = masked.argsort(descending=True)[:k].tolist()
            out[m] = [int(i) for i in topk]
        return out


# Module-level singleton
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load()
    return _engine
