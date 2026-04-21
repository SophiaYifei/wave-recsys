"""Generate error-analysis figures for the paper from data/outputs/case_studies.json.

Two figure styles:

- Style A  — L2 coherence vs L3 modality entropy scatter (3 models per case),
            with green "ideal" and orange "precision-coverage failure"
            background regions to anchor the reader.

- Style B  — N-row × 4-column text grid of each (model, modality) top-1 item,
            with per-row coloured borders.

Output: figures/error_case_*.png at dpi=200, sans-serif.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
CASES_PATH = REPO / "data" / "outputs" / "case_studies.json"
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Academic sans-serif defaults — no seaborn, no fancy cmaps.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#333",
    "axes.titlecolor": "#111",
    "axes.titleweight": "bold",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

MODEL_STYLES = {
    "popularity":  {"color": "#777777", "marker": "s", "label": "Popularity"},
    "knn":         {"color": "#2E6BAE", "marker": "o", "label": "KNN"},
    "two_tower":   {"color": "#C0392B", "marker": "^", "label": "Two-Tower"},
}

ROW_LABEL = {"popularity": "Popularity", "knn": "KNN", "two_tower": "Two-Tower"}
MODALITIES = ["book", "film", "music", "writing"]
COL_LABEL = ["Book", "Film", "Music", "Writing"]


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _load_case(case_id: str) -> Dict[str, Any]:
    cases = json.loads(CASES_PATH.read_text())
    for c in cases:
        if c["case_id"] == case_id:
            return c
    raise KeyError(f"{case_id} not found in {CASES_PATH}")


# ---------------------------------------------------------------------------
# Style A — scatter
# ---------------------------------------------------------------------------


def make_style_a(case_id: str, title: str, out_path: Path) -> Path:
    case = _load_case(case_id)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Ideal region: high entropy + high coherence (upper right)
    ax.add_patch(
        Rectangle((1.0, 0.15), 0.45, 0.15,
                  facecolor="#4CAF50", alpha=0.15, edgecolor="none", zorder=1)
    )
    ax.text((1.0 + 1.45) / 2, (0.15 + 0.30) / 2,
            "Ideal:\ndiverse & coherent",
            ha="center", va="center", fontsize=9, color="#2E7D32",
            alpha=0.8, zorder=2)

    # Failure region: low entropy + any coherence (left band)
    ax.add_patch(
        Rectangle((0, 0), 0.3, 0.30,
                  facecolor="#FF9800", alpha=0.15, edgecolor="none", zorder=1)
    )
    ax.text(0.15, 0.15,
            "Precision-coverage\nfailure",
            ha="center", va="center", fontsize=9, color="#B85C00",
            alpha=0.8, zorder=2)

    # Scatter
    for model, style in MODEL_STYLES.items():
        x = case["modality_entropy"].get(model)
        y = case["coherence_scores"].get(model)
        if x is None or y is None:
            continue
        ax.scatter(
            x, y,
            s=140,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            edgecolors="black", linewidths=0.8,
            zorder=3,
        )

    ax.set_xlim(0, 1.45)
    ax.set_ylim(0, 0.30)
    ax.set_xlabel("L3 Modality Entropy (higher = more diverse)")
    ax.set_ylabel("L2 Cross-Modal Coherence (higher = more unified)")
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Style B — item text grid
# ---------------------------------------------------------------------------


def _rank1(case: Dict[str, Any], model: str, modality: str) -> Optional[Dict[str, Any]]:
    for item in case.get("results_per_model", {}).get(model, []):
        if item.get("modality") == modality and item.get("rank") == 1:
            return item
    return None


def make_style_b(
    case_id: str,
    rows: List[str],
    title: str,
    out_path: Path,
    figsize=(12, 5),
    row_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    bottom_caption: Optional[str] = None,
) -> Path:
    case = _load_case(case_id)
    n = len(rows)
    cols = len(COL_LABEL)

    fig, ax = plt.subplots(figsize=figsize)
    # Layout:
    #   y=0..0.5   → column header row
    #   y=0.5..N+0.5 → N data rows of height 1
    #   x=-0.5..0  → row-label column
    #   x=0..4     → data cells
    ax.set_xlim(-0.55, cols + 0.02)
    ax.set_ylim(-0.05, n + 0.5 + 0.05)
    ax.invert_yaxis()
    ax.axis("off")

    # Column headers
    for j, label in enumerate(COL_LABEL):
        ax.text(j + 0.5, 0.25, label,
                ha="center", va="center",
                fontweight="bold", fontsize=11, color="#333")

    default_style = {"linewidth": 0.5, "edgecolor": "gray"}

    for i, model in enumerate(rows):
        y_top = 0.5 + i
        # Row label
        ax.text(-0.3, y_top + 0.5, ROW_LABEL.get(model, model),
                ha="center", va="center",
                fontweight="bold", fontsize=11, color="#111")
        # Cells
        style = (row_styles or {}).get(model, default_style)
        for j, mod in enumerate(MODALITIES):
            rect = Rectangle(
                (j, y_top), 1, 1,
                facecolor="white",
                linewidth=style.get("linewidth", 0.5),
                edgecolor=style.get("edgecolor", "gray"),
            )
            ax.add_patch(rect)
            item = _rank1(case, model, mod)
            if item is None:
                ax.text(j + 0.5, y_top + 0.5, "—",
                        ha="center", va="center",
                        fontsize=10, color="#999")
                continue
            title_str = _truncate(item.get("title", ""), 28)
            creator_str = _truncate(item.get("creator", ""), 25)
            # Title upper half (bold), creator lower half (muted)
            ax.text(j + 0.5, y_top + 0.38, title_str,
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#111",
                    wrap=True)
            ax.text(j + 0.5, y_top + 0.68, creator_str,
                    ha="center", va="center",
                    fontsize=8, color="#666", style="italic")

    fig.suptitle(title, fontsize=13, fontweight="bold", color="#111", y=0.98)
    if bottom_caption:
        fig.text(0.5, 0.02, bottom_caption,
                 ha="center", fontsize=10, style="italic", color="#444")

    # Tight but leave room for title + optional caption
    bottom_margin = 0.08 if bottom_caption else 0.02
    fig.subplots_adjust(left=0.03, right=0.99, top=0.89, bottom=bottom_margin)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure specs
# ---------------------------------------------------------------------------


def main() -> None:
    produced: List[tuple] = []

    # Figure 1 — Style A, modality-collapse doomscrolling query (TT entropy 0.0).
    p = make_style_a(
        case_id="case_11",
        title='Case 1: "something to stop me doomscrolling"',
        out_path=OUT_DIR / "error_case_1_modality_collapse_doomscrolling.png",
    )
    produced.append(p)

    # Figure 2 — Style B, all 3 models. Popularity row flagged in red to
    # visually anchor the failure being called out; KNN + Two-Tower stay gray.
    p = make_style_b(
        case_id="case_05",
        rows=["popularity", "knn", "two_tower"],
        title='Case 2: Popularity recommends "The Boys" for a tender existential query',
        out_path=OUT_DIR / "error_case_2_popularity_failure.png",
        figsize=(12, 5),
        row_styles={
            "popularity": {"linewidth": 2.0, "edgecolor": "#C0392B"},
            "knn":        {"linewidth": 0.5, "edgecolor": "gray"},
            "two_tower":  {"linewidth": 0.5, "edgecolor": "gray"},
        },
    )
    produced.append(p)

    # Figure 4 — Style B, KNN + Two-Tower only, colour-coded borders.
    p = make_style_b(
        case_id="case_12",
        rows=["knn", "two_tower"],
        title="Case 4: KNN matches aesthetic tags; Two-Tower drifts",
        out_path=OUT_DIR / "error_case_4_air_before_snow.png",
        figsize=(12, 4),
        row_styles={
            "knn":       {"linewidth": 2, "edgecolor": "seagreen"},
            "two_tower": {"linewidth": 2, "edgecolor": "darkorange"},
        },
        bottom_caption="Query aesthetic tags: minimalist, glass, austere, japandi",
    )
    produced.append(p)

    # Figure 5 — Style A, modality-collapse rain-on-glass case (entropy 0.199).
    p = make_style_a(
        case_id="case_07",
        title='Case 5: "something quiet and slow, like rain on glass"',
        out_path=OUT_DIR / "error_case_5_modality_collapse_rain.png",
    )
    produced.append(p)

    # Report
    print()
    print("=== saved figures ===")
    for path in produced:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            # Read PNG header to report pixel dimensions.
            import struct
            with path.open("rb") as f:
                header = f.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", header[16:24])
                print(f"  {path}  {w}x{h} px  {size_kb:.1f} KB")
            else:
                print(f"  {path}  {size_kb:.1f} KB")
        else:
            print(f"  {path}  MISSING")


if __name__ == "__main__":
    main()
