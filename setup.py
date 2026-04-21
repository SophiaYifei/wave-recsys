"""Project setup orchestrator.

Runs the end-to-end Wave pipeline (or any individual stage) via subprocess
dispatch to the existing `scripts/` modules. Intended entry point for a
fresh clone — a stranger who just ran `pip install -r requirements.txt`
should be able to reach a working system through this file alone.

Usage
-----
    python setup.py --step bootstrap   # download pre-built artifacts from HF (fastest, default)
    python setup.py --step collect     # re-collect raw data (needs API keys; see README §3)
    python setup.py --step features    # build features.npz from catalog + profiles
    python setup.py --step train       # train KNN + Two-Tower + no_intent ablation
    python setup.py --step evaluate    # 4-layer eval + case studies
    python setup.py --step experiment  # hyperparam sweep + cross-modal transfer
    python setup.py --step all         # collect -> profiles -> features -> train -> evaluate -> experiment

The `bootstrap` step is the recommended path for graders / reviewers who
only want to try the deployed-style demo: it pulls the canonical
`model.pt`, `config.pt`, `features.npz`, `catalog.jsonl`, `profiles.jsonl`,
and `weights.json` from the public HuggingFace dataset (YifeiGuo/wave-artifacts)
and skips collection + training entirely.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DEFAULT_HF_REPO = "YifeiGuo/wave-artifacts"

STEPS = ("bootstrap", "collect", "features", "train", "evaluate", "experiment", "all")


def _run(cmd: Sequence[str]) -> None:
    """Run a command, stream output, exit on non-zero."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nstep failed with exit code {result.returncode}; aborting.", file=sys.stderr)
        sys.exit(result.returncode)


def _ensure_hf_repo_env() -> None:
    if not os.environ.get("HF_REPO_ID"):
        print(
            f"HF_REPO_ID not set; defaulting to {DEFAULT_HF_REPO!r}",
            flush=True,
        )
        os.environ["HF_REPO_ID"] = DEFAULT_HF_REPO


def step_bootstrap() -> None:
    """Download pre-built artifacts from the HF dataset so the app runs immediately."""
    _ensure_hf_repo_env()
    _run([
        PYTHON, "-c",
        "from app.backend.inference import download_artifacts_if_missing; "
        "download_artifacts_if_missing()",
    ])


def step_collect() -> None:
    """Re-collect raw items for all 4 modalities (needs Spotify/TMDB/etc. keys)."""
    for source, target in (
        ("books", "600"),
        ("films", "600"),
        ("music", "1500"),
        ("writing", "1250"),
    ):
        _run([PYTHON, "scripts/collect.py", f"--source={source}", f"--target-count={target}"])


def step_features() -> None:
    """Build catalog, LLM profiles, paraphrase queries, featurized npz, and training features."""
    _run([PYTHON, "scripts/features.py", "--step=unify"])
    _run([PYTHON, "scripts/generate_profiles.py", "--step=profile"])
    _run([PYTHON, "scripts/generate_profiles.py", "--step=paraphrase"])
    _run([PYTHON, "scripts/featurize_queries.py", "--step=all"])
    _run([PYTHON, "scripts/features.py", "--step=build"])


def step_train() -> None:
    """Train KNN + Two-Tower (d64/e40, the winning sweep config) + no_intent ablation."""
    _run([PYTHON, "scripts/train.py", "--model=knn"])
    _run([PYTHON, "scripts/train.py", "--model=two_tower", "--embed-dim=64", "--max-epochs=40"])
    _run([PYTHON, "scripts/train.py", "--model=two_tower", "--ablation=no_intent"])


def step_evaluate() -> None:
    """Run the 4-layer eval + generate case_studies.json."""
    _run([PYTHON, "scripts/evaluate.py", "--model=all"])


def step_experiment() -> None:
    """Hyperparameter sweep (7 configs) + cross-modal transfer (4 holdouts) + final promote."""
    _run([PYTHON, "scripts/experiment.py", "--type=hyperparam_sweep"])
    _run([PYTHON, "scripts/experiment.py", "--type=train_final"])
    _run([PYTHON, "scripts/experiment.py", "--type=cross_modal_transfer"])


def step_all() -> None:
    """Full end-to-end pipeline. Needs all API keys (see README §3)."""
    step_collect()
    step_features()
    step_train()
    step_evaluate()
    step_experiment()


DISPATCH = {
    "bootstrap": step_bootstrap,
    "collect": step_collect,
    "features": step_features,
    "train": step_train,
    "evaluate": step_evaluate,
    "experiment": step_experiment,
    "all": step_all,
}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--step",
        choices=STEPS,
        default="bootstrap",
        help="Pipeline stage to run (default: bootstrap).",
    )
    args = parser.parse_args(argv)
    DISPATCH[args.step]()
    print(f"\nsetup: step={args.step!r} done.")


if __name__ == "__main__":
    main()
