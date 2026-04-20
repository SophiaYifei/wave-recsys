"""Train the KNN and Two-Tower models per spec §5.4.

Usage:
    python scripts/train.py --model knn
    python scripts/train.py --model two_tower [--ablation no_intent]
                                              [--holdout-modality {book,film,music,writing}]
                                              [--embed-dim 128]
                                              [--device {cpu,mps,cuda}]

Reads data/processed/features.npz (item features) and
data/processed/paraphrase_queries_featurized.npz (query features).
Paraphrase queries are split 80/10/10 stratified by modality.

In-batch negative handling (spec §7.4): queries are shuffled per epoch without
explicit same-source-item de-duplication. With ~10k queries / ~3.3k items and
batch_size=128, expected in-batch false-negative rate is ≤3/128 ≈ 2%, acceptable
for contrastive training. If this becomes a bottleneck, swap for an item-id
aware sampler.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.npz"
QUERY_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "paraphrase_queries_featurized.npz"
KNN_WEIGHTS_PATH = REPO_ROOT / "models" / "knn_weights" / "weights.json"
TT_DIR = REPO_ROOT / "models" / "two_tower"

SPLIT_SEED = 2026
TRAIN_PCT, VAL_PCT = 0.8, 0.1  # remainder -> test
DEFAULT_EMBED_DIM = 128
DEFAULT_MAX_EPOCHS = 20
TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_item_features() -> Dict[str, Any]:
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH} not found; run features.py --step=build first", file=sys.stderr)
        sys.exit(1)
    f = np.load(FEATURES_PATH, allow_pickle=True)
    return {
        "item_ids": np.array([str(x) for x in f["item_ids"]]),
        "modalities": np.array([str(x) for x in f["modalities"]]),
        "vibe": torch.from_numpy(f["vibe_embeddings"]).float(),
        "mood": torch.from_numpy(f["mood_vectors"]).float(),
        "intent": torch.from_numpy(f["intent_vectors"]).float(),
        "tag": torch.from_numpy(f["tag_onehot"]).float(),
        "modality": torch.from_numpy(f["modality_onehot"]).float(),
    }


def load_query_features() -> Dict[str, Any]:
    if not QUERY_FEATURES_PATH.exists():
        print(
            f"ERROR: {QUERY_FEATURES_PATH} not found; run featurize_queries.py first",
            file=sys.stderr,
        )
        sys.exit(1)
    f = np.load(QUERY_FEATURES_PATH, allow_pickle=True)
    return {
        "query_ids": np.array([str(x) for x in f["query_ids"]]),
        "item_ids": np.array([str(x) for x in f["item_ids"]]),
        "modalities": np.array([str(x) for x in f["modalities"]]),
        "vibe": torch.from_numpy(f["vibe_embeddings"]).float(),
        "mood": torch.from_numpy(f["mood_vectors"]).float(),
        "intent": torch.from_numpy(f["intent_vectors"]).float(),
        "tag": torch.from_numpy(f["tag_onehot"]).float(),
        "modality": torch.from_numpy(f["modality_onehot"]).float(),
    }


def build_splits(
    queries: Dict[str, Any],
    holdout_modality: Optional[str] = None,
    seed: int = SPLIT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 stratified by modality. With --holdout-modality, that modality's
    queries are entirely shifted to test (no training signal on it)."""
    rng = np.random.default_rng(seed)
    by_mod: Dict[str, List[int]] = {}
    for i, m in enumerate(queries["modalities"]):
        by_mod.setdefault(m, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for m, idxs in by_mod.items():
        idxs_arr = np.array(idxs)
        rng.shuffle(idxs_arr)
        if holdout_modality is not None and m == holdout_modality:
            test_idx.extend(idxs_arr.tolist())
            continue
        n = len(idxs_arr)
        n_tr = int(TRAIN_PCT * n)
        n_va = int(VAL_PCT * n)
        train_idx.extend(idxs_arr[:n_tr].tolist())
        val_idx.extend(idxs_arr[n_tr : n_tr + n_va].tolist())
        test_idx.extend(idxs_arr[n_tr + n_va :].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def build_item_id_to_idx(items: Dict[str, Any]) -> Dict[str, int]:
    return {iid: i for i, iid in enumerate(items["item_ids"])}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _cos_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    An = F.normalize(A, dim=-1)
    Bn = F.normalize(B, dim=-1)
    return An @ Bn.t()


def _jaccard_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise Jaccard between binary one-hot rows (A: B1,T) (B: B2,T) -> (B1,B2)."""
    Ab = (A > 0).float()
    Bb = (B > 0).float()
    inter = Ab @ Bb.t()
    a_card = Ab.sum(dim=-1, keepdim=True)
    b_card = Bb.sum(dim=-1, keepdim=True).t()
    union = (a_card + b_card - inter).clamp(min=1e-8)
    return inter / union


class WeightedKNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_logits = nn.Parameter(torch.zeros(5))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.weight_logits, dim=0)

    def score_matrix(self, q: Dict[str, torch.Tensor], i: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.weights()
        s_v = _cos_matrix(q["vibe"], i["vibe"])
        s_m = _cos_matrix(q["mood"], i["mood"])
        s_i = _cos_matrix(q["intent"], i["intent"])
        s_t = _jaccard_matrix(q["tag"], i["tag"])
        s_mod = _cos_matrix(q["modality"], i["modality"])
        return w[0] * s_v + w[1] * s_m + w[2] * s_i + w[3] * s_t + w[4] * s_mod


def _build_mlp(in_dim: int, out_dim: int, hidden=(512, 256), dropout: float = 0.2) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, out_dim)]  # final projection, no ReLU/Dropout
    return nn.Sequential(*layers)


class QueryTower(nn.Module):
    def __init__(self, output_dim: int = DEFAULT_EMBED_DIM, use_intent: bool = True):
        super().__init__()
        self.use_intent = use_intent
        in_dim = 384 + 12 + (7 if use_intent else 0)
        self.mlp = _build_mlp(in_dim, output_dim)

    def forward(self, vibe: torch.Tensor, mood: torch.Tensor, intent: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [vibe, mood]
        if self.use_intent:
            parts.append(intent)
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, output_dim: int = DEFAULT_EMBED_DIM, use_intent: bool = True):
        super().__init__()
        self.use_intent = use_intent
        in_dim = 384 + 12 + (7 if use_intent else 0) + 24 + 4
        self.mlp = _build_mlp(in_dim, output_dim)

    def forward(
        self,
        vibe: torch.Tensor,
        mood: torch.Tensor,
        intent: Optional[torch.Tensor],
        tag: torch.Tensor,
        modality: torch.Tensor,
    ) -> torch.Tensor:
        parts = [vibe, mood]
        if self.use_intent:
            parts.append(intent)
        parts.extend([tag, modality])
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        use_intent: bool = True,
        embed_dim: int = DEFAULT_EMBED_DIM,
        temperature: float = TEMPERATURE,
    ):
        super().__init__()
        self.use_intent = use_intent
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.query_tower = QueryTower(embed_dim, use_intent)
        self.item_tower = ItemTower(embed_dim, use_intent)

    def encode_query(self, q: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.query_tower(
            q["vibe"], q["mood"], q.get("intent") if self.use_intent else None
        )

    def encode_item(self, i: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.item_tower(
            i["vibe"],
            i["mood"],
            i.get("intent") if self.use_intent else None,
            i["tag"],
            i["modality"],
        )

    def info_nce_loss(self, q_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        logits = q_emb @ i_emb.t() / self.temperature
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    def __init__(
        self,
        q_indices: np.ndarray,
        queries: Dict[str, Any],
        item_id_to_idx: Dict[str, int],
    ):
        self.q_indices = q_indices
        self.queries = queries
        self.item_id_to_idx = item_id_to_idx

    def __len__(self) -> int:
        return len(self.q_indices)

    def __getitem__(self, i: int) -> Tuple[int, int]:
        qi = int(self.q_indices[i])
        item_id = str(self.queries["item_ids"][qi])
        if item_id not in self.item_id_to_idx:
            raise KeyError(
                f"PairDataset: item_id {item_id!r} (query index {qi}) not found in "
                "item_id_to_idx. Check that features.npz and paraphrase_queries_featurized.npz "
                "were built from the same catalog."
            )
        ii = self.item_id_to_idx[item_id]
        return qi, ii


def _batch_from_indices(
    q_indices: torch.Tensor,
    i_indices: torch.Tensor,
    queries: Dict[str, Any],
    items: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    q = {k: queries[k][q_indices].to(device) for k in ("vibe", "mood", "intent", "tag", "modality")}
    i = {k: items[k][i_indices].to(device) for k in ("vibe", "mood", "intent", "tag", "modality")}
    return q, i


def _make_q_batch(queries: Dict[str, Any], idx: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: queries[k][idx].to(device) for k in ("vibe", "mood", "intent", "tag", "modality")}


def _make_items_on_device(items: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: items[k].to(device) for k in ("vibe", "mood", "intent", "tag", "modality")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    score_matrix: torch.Tensor,
    gold_item_indices: torch.Tensor,
    k: int = 10,
) -> Dict[str, float]:
    topk = score_matrix.argsort(dim=1, descending=True)[:, :k]
    ndcg_sum = 0.0
    p_sum = 0.0
    map_sum = 0.0
    Q = score_matrix.size(0)
    for q in range(Q):
        gold = gold_item_indices[q].item()
        row = topk[q].tolist()
        if gold in row:
            pos = row.index(gold)
            ndcg_sum += 1.0 / np.log2(pos + 2)
            p_sum += 1.0
            map_sum += 1.0 / (pos + 1)
    return {
        f"NDCG@{k}": ndcg_sum / Q,
        f"P@{k}": p_sum / Q,
        f"MAP@{k}": map_sum / Q,
    }


# ---------------------------------------------------------------------------
# Train KNN
# ---------------------------------------------------------------------------


def train_knn(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"knn: device={device}")

    items = load_item_features()
    queries = load_query_features()
    item_id_to_idx = build_item_id_to_idx(items)

    train_idx, val_idx, test_idx = build_splits(queries)
    print(f"knn: split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    dataset = PairDataset(train_idx, queries, item_id_to_idx)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    model = WeightedKNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    items_dev = _make_items_on_device(items, device)
    val_q = _make_q_batch(queries, val_idx, device)
    val_gold = torch.tensor(
        [item_id_to_idx[str(queries["item_ids"][qi])] for qi in val_idx], dtype=torch.long
    )

    best_val_ndcg = -1.0
    best_epoch = -1
    best_weights: Optional[np.ndarray] = None
    patience = 5
    bad_epochs = 0

    for epoch in range(30):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for q_indices, i_indices in loader:
            q, i = _batch_from_indices(q_indices, i_indices, queries, items, device)
            scores = model.score_matrix(q, i)  # (B, B)
            labels = torch.arange(scores.size(0), device=device)
            loss = F.cross_entropy(scores, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_scores = model.score_matrix(val_q, items_dev).cpu()
        val_metrics = compute_retrieval_metrics(val_scores, val_gold, k=10)
        val_ndcg = val_metrics["NDCG@10"]

        print(
            f"  epoch {epoch + 1:2d} train_loss={train_loss:.4f} "
            f"val_NDCG@10={val_ndcg:.4f} val_P@10={val_metrics['P@10']:.4f} "
            f"weights={model.weights().detach().cpu().numpy().round(3).tolist()} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch + 1
            best_weights = model.weights().detach().cpu().numpy().copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch + 1} (patience={patience} exhausted)")
                break

    assert best_weights is not None
    KNN_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KNN_WEIGHTS_PATH.open("w") as f:
        json.dump(
            {
                "w_vibe": float(best_weights[0]),
                "w_mood": float(best_weights[1]),
                "w_intent": float(best_weights[2]),
                "w_tag": float(best_weights[3]),
                "w_modality": float(best_weights[4]),
                "best_epoch": best_epoch,
                "best_val_ndcg10": best_val_ndcg,
            },
            f,
            indent=2,
        )

    # Restore best weights into model for test-set metrics (invert softmax).
    best_w_t = torch.from_numpy(best_weights).float().clamp(min=1e-12)
    logits = torch.log(best_w_t)
    model.weight_logits.data = (logits - logits.mean()).to(device)

    test_q = _make_q_batch(queries, test_idx, device)
    test_gold = torch.tensor(
        [item_id_to_idx[str(queries["item_ids"][qi])] for qi in test_idx], dtype=torch.long
    )
    with torch.no_grad():
        test_scores = model.score_matrix(test_q, items_dev).cpu()
    test_m10 = compute_retrieval_metrics(test_scores, test_gold, k=10)
    test_m5 = compute_retrieval_metrics(test_scores, test_gold, k=5)

    print(f"knn: saved weights to {KNN_WEIGHTS_PATH}")
    print(f"knn: best weights (softmax) = {best_weights.round(4).tolist()}")
    print(f"knn: best_epoch={best_epoch}, best_val_NDCG@10={best_val_ndcg:.4f}")
    print(
        f"knn: test NDCG@10={test_m10['NDCG@10']:.4f} "
        f"P@10={test_m10['P@10']:.4f} MAP@10={test_m10['MAP@10']:.4f} "
        f"NDCG@5={test_m5['NDCG@5']:.4f} P@5={test_m5['P@5']:.4f}"
    )


# ---------------------------------------------------------------------------
# Train Two-Tower
# ---------------------------------------------------------------------------


def train_two_tower(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    use_intent = args.ablation != "no_intent"
    holdout = args.holdout_modality
    embed_dim = args.embed_dim
    max_epochs = args.max_epochs

    print(
        f"two_tower: device={device} use_intent={use_intent} "
        f"holdout={holdout} embed_dim={embed_dim} max_epochs={max_epochs}"
    )

    items = load_item_features()
    queries = load_query_features()
    item_id_to_idx = build_item_id_to_idx(items)

    train_idx, val_idx, test_idx = build_splits(queries, holdout_modality=holdout)
    print(f"two_tower: split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    dataset = PairDataset(train_idx, queries, item_id_to_idx)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    model = TwoTower(use_intent=use_intent, embed_dim=embed_dim, temperature=TEMPERATURE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    items_dev = _make_items_on_device(items, device)
    val_q = _make_q_batch(queries, val_idx, device)
    val_gold = torch.tensor(
        [item_id_to_idx[str(queries["item_ids"][qi])] for qi in val_idx], dtype=torch.long
    )

    best_val_ndcg = -1.0
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 5
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for q_indices, i_indices in loader:
            q, i = _batch_from_indices(q_indices, i_indices, queries, items, device)
            q_emb = model.encode_query(q)
            i_emb = model.encode_item(i)
            loss = model.info_nce_loss(q_emb, i_emb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            all_i_emb = model.encode_item(items_dev)  # (N, D)
            q_emb = model.encode_query(val_q)         # (Q, D)
            val_scores = (q_emb @ all_i_emb.t()).cpu()
        val_metrics = compute_retrieval_metrics(val_scores, val_gold, k=10)
        val_ndcg = val_metrics["NDCG@10"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_NDCG@10": val_ndcg,
            "val_P@10": val_metrics["P@10"],
        })
        print(
            f"  epoch {epoch + 1:2d} train_loss={train_loss:.4f} "
            f"val_NDCG@10={val_ndcg:.4f} val_P@10={val_metrics['P@10']:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch + 1} (patience={patience} exhausted)")
                break

    # File naming per spec §5.4:
    #   base:            model.pt / config.pt
    #   ablation:        model_no_intent.pt
    #   holdout:         model_holdout_{modality}.pt
    #   embed_dim sweep: model_d{N}.pt  (G1 variants live alongside)
    suffix_parts = []
    if not use_intent:
        suffix_parts.append("no_intent")
    if holdout:
        suffix_parts.append(f"holdout_{holdout}")
    if embed_dim != DEFAULT_EMBED_DIM:
        suffix_parts.append(f"d{embed_dim}")
    if max_epochs != DEFAULT_MAX_EPOCHS:
        suffix_parts.append(f"e{max_epochs}")
    stem = "model" + ("_" + "_".join(suffix_parts) if suffix_parts else "")
    model_path = TT_DIR / f"{stem}.pt"
    config_path = TT_DIR / "config.pt" if not suffix_parts else TT_DIR / f"{stem}_config.pt"

    TT_DIR.mkdir(parents=True, exist_ok=True)
    assert best_state is not None
    torch.save(best_state, model_path)
    torch.save(
        {
            "embed_dim": embed_dim,
            "temperature": TEMPERATURE,
            "use_intent": use_intent,
            "holdout_modality": holdout,
            "best_epoch": best_epoch,
            "best_val_NDCG@10": best_val_ndcg,
            "history": history,
        },
        config_path,
    )

    model.load_state_dict(best_state)
    model.eval()
    test_q = _make_q_batch(queries, test_idx, device)
    test_gold = torch.tensor(
        [item_id_to_idx[str(queries["item_ids"][qi])] for qi in test_idx], dtype=torch.long
    )
    with torch.no_grad():
        all_i_emb = model.encode_item(items_dev)
        q_emb = model.encode_query(test_q)
        test_scores = (q_emb @ all_i_emb.t()).cpu()
    test_m10 = compute_retrieval_metrics(test_scores, test_gold, k=10)
    test_m5 = compute_retrieval_metrics(test_scores, test_gold, k=5)

    print(f"two_tower: saved {model_path}")
    print(f"two_tower: saved config to {config_path}")
    print(f"two_tower: best_epoch={best_epoch} best_val_NDCG@10={best_val_ndcg:.4f}")
    print(
        f"two_tower: test NDCG@10={test_m10['NDCG@10']:.4f} "
        f"P@10={test_m10['P@10']:.4f} MAP@10={test_m10['MAP@10']:.4f} "
        f"NDCG@5={test_m5['NDCG@5']:.4f} P@5={test_m5['P@5']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KNN or Two-Tower.")
    parser.add_argument("--model", required=True, choices=["knn", "two_tower"])
    parser.add_argument("--ablation", choices=["no_intent"], default=None)
    parser.add_argument(
        "--holdout-modality",
        choices=["book", "film", "music", "writing"],
        default=None,
    )
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Default cpu; for small tensors on this dataset CPU is competitive.",
    )
    args = parser.parse_args()
    torch.manual_seed(SPLIT_SEED)

    if args.model == "knn":
        if args.ablation or args.holdout_modality:
            print("knn: --ablation/--holdout-modality ignored for knn", file=sys.stderr)
        train_knn(args)
    else:
        train_two_tower(args)


if __name__ == "__main__":
    main()
