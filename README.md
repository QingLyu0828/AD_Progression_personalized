# PROMISE-AD

**PROgression-aware MultI-horizon Survival Estimation for Alzheimer's Disease.**

This release contains the PROMISE-AD model, training entry point, and standalone inference entry point for CN-to-MCI and MCI-to-AD progression modeling. It starts from prepared longitudinal visit tables. Raw-data cohort construction, baseline models, ablations, experiment orchestration, datasets, and trained weights are outside this six-file release.

## Files and responsibilities

| File | Purpose |
|---|---|
| `train.py` | Defines the PROMISE-AD model and its complete training routine: training-only preprocessing, longitudinal visit sequences, tokenization, temporal Transformer, representation fusion, progression score, mixture hazards, five loss terms, checkpoint selection, prediction, and evaluation. Run this file to train a model for one task and one seed. |
| `inference.py` | Loads a checkpoint, fitted preprocessor, and configuration from one training run. Applies the existing preprocessing to new visit histories and exports one row of predictions per subject without requiring outcome labels. It does not train or recalibrate a model. |
| `ad_progression_eval_utils.py` | Shared evaluation functions imported by `train.py`: censoring-aware horizon labels, validation-only isotonic/Platt calibration, validation threshold selection, classification metrics, C-index, time-dependent AUC, Brier/IBS, and risk-group statistics. This is a library module, not a separate training command. Some survival metrics require optional dependencies. |
| `requirements.txt` | Core Python dependencies for training and inference. |
| `.gitignore` | Excludes common data, checkpoint, preprocessing, and generated-output files from Git. |
| `README.md` | File descriptions, installation, input requirements, execution examples, and correspondence with the manuscript. |

The main components inside `train.py` are:

- `LongitudinalVisitPreprocessor`: fits medians, means, standard deviations, and categorical vocabularies on training data; constructs values, changes, slopes, masks, and visit timing.
- `ADProgressionDataset` and `collate_visit_sequences`: construct and pad subject-level visit sequences.
- `VisitTokenizer`: combines numeric/missingness, categorical, and timing projections.
- `ProgressionAwareMultiHorizonSurvivalModel`: applies visit dropout, a Transformer, global/attention-pooled/latest-visit fusion, progression scoring, and probability-space hazard mixing.
- `total_loss` and its component functions: combine survival likelihood, horizon focal loss, progression ranking, hazard smoothness, and gate balance.
- `train_model` and `ModelEMA`: optimize with AdamW and select an EMA checkpoint using validation loss.
- `predict`: converts hazards to raw cumulative risks at the configured horizons.

## Installation

Python 3.11 is recommended. Use an isolated environment:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, install the PyTorch build appropriate for your CUDA environment. For the additional survival metrics used in the paper:

```bash
python -m pip install scikit-survival==0.27.0 'lifelines>=0.29,<0.31'
```

Without these optional packages, core training and inference remain available, but metrics such as IPCW Brier/IBS or log-rank statistics may be unavailable. Inspect the generated metrics and calibration-status fields rather than assuming every metric was computed.

## Input data

Train the two tasks separately. Supply three CSV files with subject-disjoint training, validation, and test sets; each row represents one eligible historical visit.

| Default column | Meaning | Training | Inference |
|---|---|---|---|
| `RID` | Subject identifier | Required | Required |
| `EXAMDATE` | Visit date and ordering | Supply | Required |
| `time_interval_to_first_visit_years` | Elapsed time from the first retained visit | Supply | Supply using the same convention; dates are a fallback |
| `index_date` | Event or control pseudo-index date | Supply for the pre-index check | Not required or used |
| `time_to_index_years` | Survival target under the study's time-origin definition | Required | Not required or used |
| `label` | Event indicator: 1 for conversion, 0 for censoring | Required | Not required or used |
| Selected predictor columns | Numeric and categorical measurements | Required | All columns fitted during training must be present; missing values may be blank |

Prepare the study's cohort, targets, feature schema, and frozen 70/15/15 subject split **before** calling this package. The manuscript specifies at least two retained pre-index visits, row missingness no greater than 10%, feature missingness no greater than 10%, and dominant-mode frequency no greater than 99.5%. Convert TADPOLE `-4` missing-value codes to missing values upstream and exclude diagnostic/outcome-derived predictors. The local revision's primary feature policy also excludes `FLDSTRENG`.

All supplied rows must already satisfy the pre-index rule: `train.py` fits preprocessing statistics before its per-subject safety filter, so that filter alone cannot prevent post-index rows from affecting fitted statistics. Feature screening and cohort matching are not implemented by this minimal package. The single-CSV convenience split is also not the paper's frozen split: its defaults are 60/20/20 and its stratification uses the event label only. Use the three pre-split inputs below for manuscript-oriented runs.

## Training with the reported settings

The following example sets those parameters explicitly and repeats training over seeds 0-4 on the same supplied split.

```bash
for seed in 0 1 2 3 4; do
  python train.py \
    --train_csv_path data/mci_to_ad_train_selected_features.csv \
    --val_csv_path data/mci_to_ad_validation_selected_features.csv \
    --test_csv_path data/mci_to_ad_test_selected_features.csv \
    --out_dir outputs/mci_to_ad/seed_${seed} \
    --categorical_cols PTGENDER APOE4 \
    --horizons 1 2 3 5 \
    --bins 0.5 1 1.5 2 2.5 3 4 5 \
    --min_visits 2 --max_visits 32 \
    --d_model 128 --cat_emb_dim 16 --n_heads 4 --n_layers 2 \
    --dim_feedforward 256 --dropout 0.25 --n_experts 4 \
    --visit_dropout 0.15 \
    --batch_size 32 --epochs 180 --lr 0.0002 --weight_decay 0.0003 \
    --auto_event_weight --max_event_weight 5 \
    --auto_horizon_pos_weights --max_horizon_pos_weight 8 \
    --lambda_horizon 0.40 --lambda_progression 0.10 \
    --lambda_smooth 0.01 --lambda_gate_balance 0.01 \
    --horizon_focal_gamma 1.5 \
    --selection_metric val_loss --min_delta 0.00001 --patience 30 \
    --ema_decay 0.995 --lr_scheduler plateau \
    --lr_plateau_factor 0.5 --lr_plateau_patience 6 --min_lr 0.000001 \
    --calibration_method isotonic --threshold_strategy youden \
    --num_workers 0 --device cuda --seed "${seed}"
done
```

For CN-to-MCI, use the corresponding input files and a separate output directory. Use `--device cpu` for CPU execution. For a single run, execute the command once with a chosen seed. Gradient clipping at norm 1.0 is implemented in the training configuration. The weight caps and plateau-scheduler settings above are supported by the local revision configuration; not all are stated explicitly in the manuscript. See `python train.py --help` for all options.

Training writes `best_model.pt`, `visit_preprocessor.pkl`, `config.json`, `feature_columns.json`, training history, and predictions. `predictions_*.csv` and `metrics_*.json` contain the native raw-prediction results. The `proposed_survival_*` tables contain the standardized evaluation: eligible horizon metrics use validation-set calibration and validation-selected thresholds; C-index and integrated survival-curve metrics use raw survival predictions. Fitted calibration objects are not exported by this implementation.

## Inference

Keep these three artifacts from the **same training run**:

- `best_model.pt`: selected model weights.
- `visit_preprocessor.pkl`: fitted feature schema, imputation/scaling statistics, and categorical vocabularies.
- `config.json`: architecture, hazard bins, horizons, and training settings.

```bash
python inference.py \
  --model-dir outputs/mci_to_ad/seed_0 \
  --input-csv data/new_subject_visits.csv \
  --output-csv outputs/predictions.csv \
  --device cpu
```

The input must contain only the intended observation window, using the same predictor units, categorical encoding, and time convention as training. The inference entry point does not truncate histories using an event or pseudo-index date; it ignores outcome columns and uses the supplied history. It reuses the saved preprocessing without fitting on new subjects and rejects missing predictor columns or unusable subjects.

Each subject receives `risk_1y`, `risk_2y`, `risk_3y`, and `risk_5y` when those horizons were trained, plus `risk_max_bin`, `progression_score`, and `hazard_bin*`. These are **uncalibrated model outputs**, consistent with `predictions_test.csv`, not the calibrated probabilities used by the standardized horizon evaluation. The horizons retain the study's original time origin; they are not automatically redefined as years after the latest visit. This entry point supports loading and scoring an individual trained model, not averaging five seeds.
