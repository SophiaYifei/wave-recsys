# `data/outputs/`

Committed artifacts produced by `scripts/evaluate.py` and `scripts/experiment.py`. Referenced by the report and the top-level README. Every file here reflects the **final model** (Two-Tower, `embed_dim=64`, `max_epochs=40`).

| File | Produced by | What it holds |
|---|---|---|
| `eval_results.json` | `scripts/evaluate.py --model=all` | 4-layer eval across `popularity` / `knn` / `two_tower` / `two_tower_no_intent`: retrieval (P/R/NDCG/MAP at K∈{5,10}), independent-encoder coherence, modality-coverage entropy, `openai/gpt-5.4-nano` judge scores on 50 random test queries × 3 models. |
| `case_studies.json` | `scripts/evaluate.py` (same run) | 10 qualitative case studies sampled from 20 probe queries across four categories: `modality_collapse`, `low_coherence`, `high_cross_modal_coherence`, `knn_two_tower_disagreement`. Raw material for the Error Analysis section of the report. |
| `hyperparam_sweep.json` | `scripts/experiment.py --type=hyperparam_sweep` | Test metrics for seven configurations: `embed_dim ∈ {32, 64, 128, 256, 512}` at `max_epochs=40`, plus `embed_dim=128` at `max_epochs ∈ {20, 60}`. Best config (d64/e40) is promoted as the canonical `model.pt`. |
| `hyperparam_sweep.png` | same | `embed_dim` vs `NDCG@10` at `max_epochs=40` — the plot that lives in the report's Hyperparameter chapter. |
| `cross_modal_transfer.json` | `scripts/experiment.py --type=cross_modal_transfer` | Baseline NDCG@10 (`d128/e20`) vs four holdout-modality Two-Towers trained at `max_epochs=40` and evaluated on the full test split. Documents the 30–51% transfer gap reported in README limitations. |
| `final_model_info.json` | `scripts/experiment.py --type=train_final` | Metadata for the promoted canonical model (winning sweep config, best epoch, the intermediate filename it was copied from). |

`.gitkeep` marks the directory so it stays checked-in when the files are regenerated.
