#!/usr/bin/env python
"""
Progression-aware multihorizon survival model for longitudinal AD progression.

PROMISE-AD combines masked visit dropout, global/attention-pooled/latest-visit
representation fusion, probability-space mixture hazards, weighted focal horizon
loss, hazard smoothness, mixture gate balancing, EMA checkpoints, and LR scheduling.

Tasks implemented:
1) Convert pre-index longitudinal tabular visits into model-ready visit inputs and learned visit tokens.
2) Build a progression-aware multihorizon survival model with:
   - visit tokenizer: numeric values + change features + missingness + categorical embeddings + time gaps
   - temporal Transformer encoder
   - progression score head
   - subtype-aware mixture survival head producing discrete-time hazards and horizon risks
3) Train, validate, and test the model with leakage-safe subject-level splitting.

Examples:

A) Use one unsplit CSV and let this script create subject-level train/val/test splits:
python train.py \
  --csv_path /mnt/data/cn_to_mci_selected_features.csv \
  --out_dir /mnt/data/pamsm_results \
  --horizons 2 3 5 \
  --bins 0.5 1 1.5 2 2.5 3 4 5 \
  --epochs 80 --batch_size 32 --lr 1e-4

B) Use existing subject-level split CSV files:
python train.py \
  --train_csv_path /path/to/train.csv \
  --val_csv_path /path/to/val.csv \
  --test_csv_path /path/to/test.csv \
  --out_dir /mnt/data/pamsm_results_presplit \
  --horizons 2 3 5 \
  --bins 0.5 1 1.5 2 2.5 3 4 5 \
  --epochs 80 --batch_size 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        brier_score_loss,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from ad_progression_eval_utils import (
    normalize_list_arg,
    standard_evaluate_survival_predictions,
    hazards_from_prediction_dataframe,
)


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class DataConfig:
    id_col: str = "RID"
    date_col: str = "EXAMDATE"
    time_col: str = "time_interval_to_first_visit_years"
    duration_col: str = "time_to_index_years"
    event_col: str = "label"
    index_date_col: str = "index_date"
    min_visits: int = 1
    max_visits: int = 32
    keep_recent_if_too_long: bool = True
    strict_preindex_filter: bool = True
    use_time_derived_features: bool = True
    categorical_cols: Optional[List[str]] = None
    numeric_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None


@dataclass
class ModelConfig:
    d_model: int = 128
    cat_emb_dim: int = 16
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.15
    n_experts: int = 3
    max_visits: int = 32
    visit_dropout: float = 0.10


@dataclass
class TrainConfig:
    horizons: Tuple[float, ...] = (2.0, 3.0, 5.0)
    bins: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    batch_size: int = 32
    epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-4
    event_weight: float = 2.0
    lambda_horizon: float = 0.25
    lambda_progression: float = 0.10
    lambda_smooth: float = 0.02
    lambda_gate_balance: float = 0.01
    horizon_pos_weights: Tuple[float, ...] = ()
    horizon_focal_gamma: float = 1.0
    grad_clip: float = 1.0
    patience: int = 15
    min_delta: float = 1e-5
    selection_metric: str = "val_loss"
    lr_scheduler: str = "plateau"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 5
    min_lr: float = 1e-6
    ema_decay: float = 0.995
    num_workers: int = 0
    seed: int = 42


# -----------------------------
# Preprocessor and visit-sequence builder
# -----------------------------

class LongitudinalVisitPreprocessor:
    """
    Fits train-only statistics and vocabularies, then converts a subject's pre-index rows
    into arrays used by the visit tokenizer.

    Numeric part per visit:
        [z-scored value, delta-from-first-visit, slope-from-first-visit]
    plus a matching missingness mask for each part.

    Categorical part per visit:
        integer IDs for learned embeddings.

    Time part per visit:
        absolute time since first visit and delta-time since previous visit, in years.
    """

    DEFAULT_EXCLUDE_SUBSTRINGS = (
        "matched_",
        "matching_distance",
    )

    DEFAULT_EXCLUDE_COLS = {
        "subset", "split", "fold", "PTID", "label_name", "index_type", "baseline_date",
        "time_to_index_days", "time_to_index_years", "time_to_event", "event_time",
        "matched_target_days", "matched_reference_subject_id",
        "matched_reference_time_to_index_days", "matched_reference_time_to_index_years",
        "VISCODE", "Month", "time_interval_to_first_visit_days",
        # Leakage/design variables. These should not be used as predictors for
        # fair comparison with the baseline scripts.
        "DX", "DXCHANGE", "diagnosis", "Diagnosis", "baseline_stage", "visit_stage",
    }

    SUGGESTED_CATEGORICAL = {
        # Keep only non-leaky baseline attributes by default. Diagnosis/stage
        # columns are excluded unless the user explicitly overrides --drop_cols
        # and --categorical_cols for a diagnostic-information sensitivity analysis.
        "PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY", "APOE4",
    }

    def __init__(self, config: DataConfig):
        self.config = config
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.numeric_median: Optional[pd.Series] = None
        self.numeric_mean: Optional[pd.Series] = None
        self.numeric_std: Optional[pd.Series] = None
        self.vocabs: Dict[str, Dict[str, int]] = {}

    def _is_excluded(self, col: str) -> bool:
        cfg = self.config
        excluded = set(self.DEFAULT_EXCLUDE_COLS)
        excluded.update({cfg.id_col, cfg.date_col, cfg.duration_col, cfg.event_col})
        if cfg.index_date_col:
            excluded.add(cfg.index_date_col)
        if cfg.time_col:
            excluded.add(cfg.time_col)
        if cfg.drop_cols:
            excluded.update(cfg.drop_cols)
        if col in excluded:
            return True
        return any(s in col for s in self.DEFAULT_EXCLUDE_SUBSTRINGS)

    @staticmethod
    def _clean_cat_value(v: Any) -> str:
        if pd.isna(v):
            return "__MISSING__"
        return str(v)

    def infer_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if self.config.numeric_cols is not None:
            numeric = [c for c in self.config.numeric_cols if c in df.columns]
        else:
            numeric = []
            for c in df.columns:
                if self._is_excluded(c):
                    continue
                if pd.api.types.is_numeric_dtype(df[c]) and c not in self.SUGGESTED_CATEGORICAL:
                    numeric.append(c)

        if self.config.categorical_cols is not None:
            categorical = [c for c in self.config.categorical_cols if c in df.columns]
        else:
            categorical = []
            for c in df.columns:
                if self._is_excluded(c):
                    continue
                if c in self.SUGGESTED_CATEGORICAL or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
                    categorical.append(c)

        # Prevent overlap.
        categorical_set = set(categorical)
        numeric = [c for c in numeric if c not in categorical_set]
        return numeric, categorical

    def fit(self, train_df: pd.DataFrame) -> "LongitudinalVisitPreprocessor":
        self.numeric_cols, self.categorical_cols = self.infer_columns(train_df)
        if len(self.numeric_cols) == 0:
            raise ValueError("No numeric feature columns were inferred. Please pass --numeric_cols.")

        num = train_df[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        self.numeric_median = num.median(axis=0).fillna(0.0)
        imputed = num.fillna(self.numeric_median)
        self.numeric_mean = imputed.mean(axis=0).fillna(0.0)
        self.numeric_std = imputed.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

        self.vocabs = {}
        for col in self.categorical_cols:
            values = train_df[col].map(self._clean_cat_value).astype(str).unique().tolist()
            values = sorted([v for v in values if v not in {"__MISSING__", "__PAD__", "__UNK__"}])
            vocab = {"__PAD__": 0, "__UNK__": 1, "__MISSING__": 2}
            for v in values:
                if v not in vocab:
                    vocab[v] = len(vocab)
            self.vocabs[col] = vocab
        return self

    def _get_times(self, g: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        if cfg.time_col in g.columns:
            times = pd.to_numeric(g[cfg.time_col], errors="coerce").to_numpy(dtype=np.float32)
            if np.all(np.isnan(times)):
                times = None
            else:
                # Fill occasional missing times from dates if possible; otherwise forward fill.
                if np.any(np.isnan(times)):
                    s = pd.Series(times).ffill().bfill().fillna(0.0)
                    times = s.to_numpy(dtype=np.float32)
        else:
            times = None

        if times is None:
            dates = pd.to_datetime(g[cfg.date_col], errors="coerce")
            if dates.isna().all():
                times = np.arange(len(g), dtype=np.float32)
            else:
                t0 = dates.min()
                times = ((dates - t0).dt.days / 365.25).fillna(0.0).to_numpy(dtype=np.float32)
        times = np.maximum(times - np.nanmin(times), 0.0).astype(np.float32)
        return times

    def transform_group(self, g: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.config
        if cfg.date_col in g.columns:
            g = g.copy()
            g[cfg.date_col] = pd.to_datetime(g[cfg.date_col], errors="coerce")
            g = g.sort_values(cfg.date_col)

        # Optional safety filter. This is useful if a source file accidentally includes index/post-index visits.
        if cfg.strict_preindex_filter and cfg.index_date_col in g.columns and cfg.date_col in g.columns:
            idx_date = pd.to_datetime(g[cfg.index_date_col].dropna().iloc[0], errors="coerce") if g[cfg.index_date_col].notna().any() else pd.NaT
            if pd.notna(idx_date):
                g = g[g[cfg.date_col] < idx_date]

        if len(g) == 0:
            raise ValueError("A subject has no visits after pre-index filtering.")

        if len(g) > cfg.max_visits:
            g = g.iloc[-cfg.max_visits:] if cfg.keep_recent_if_too_long else g.iloc[:cfg.max_visits]

        # Numeric values and masks.
        num_raw = g[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
        obs_mask = (~num_raw.isna()).astype(np.float32).to_numpy()
        num_imputed = num_raw.fillna(self.numeric_median)
        num_scaled = ((num_imputed - self.numeric_mean) / self.numeric_std).to_numpy(dtype=np.float32)

        observed_times = self._get_times(g)
        if cfg.use_time_derived_features:
            times = observed_times
            delta_t = np.diff(times, prepend=times[0]).astype(np.float32)
            delta_t = np.maximum(delta_t, 0.0)
        else:
            # Timing-sensitivity condition: retain visit order and raw/delta values,
            # but remove the explicit absolute/gap token and every time-normalized
            # slope.  Using zeros here avoids leaking visit spacing through a
            # fallback date-derived time channel.
            times = np.zeros(len(g), dtype=np.float32)
            delta_t = np.zeros(len(g), dtype=np.float32)

        # Change features: delta and slope from first available visit.
        baseline = num_scaled[0:1, :]
        delta_from_first = num_scaled - baseline
        if cfg.use_time_derived_features:
            denom = np.maximum(times.reshape(-1, 1), 1e-3)
            slope_from_first = delta_from_first / denom
            slope_from_first[0, :] = 0.0
        else:
            slope_from_first = np.zeros_like(delta_from_first, dtype=np.float32)

        base_mask = obs_mask[0:1, :]
        delta_mask = obs_mask * base_mask
        x_num = np.concatenate([num_scaled, delta_from_first, slope_from_first], axis=1).astype(np.float32)
        x_num_mask = np.concatenate([obs_mask, delta_mask, delta_mask], axis=1).astype(np.float32)

        # Categorical IDs.
        if len(self.categorical_cols) > 0:
            x_cat = []
            for col in self.categorical_cols:
                vocab = self.vocabs[col]
                ids = [vocab.get(self._clean_cat_value(v), vocab["__UNK__"]) for v in g[col].tolist()]
                x_cat.append(ids)
            x_cat = np.asarray(x_cat, dtype=np.int64).T
        else:
            x_cat = np.zeros((len(g), 0), dtype=np.int64)

        duration = float(pd.to_numeric(g[cfg.duration_col], errors="coerce").dropna().iloc[0])
        event = int(pd.to_numeric(g[cfg.event_col], errors="coerce").dropna().iloc[0])
        rid = g[cfg.id_col].iloc[0]

        return {
            "rid": rid,
            "x_num": x_num,
            "x_num_mask": x_num_mask,
            "x_cat": x_cat,
            "times": times.astype(np.float32),
            "delta_t": delta_t.astype(np.float32),
            "duration": np.float32(duration),
            "event": np.int64(event),
            "n_visits": len(g),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "LongitudinalVisitPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


class ADProgressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, subject_ids: List[Any], preprocessor: LongitudinalVisitPreprocessor):
        self.preprocessor = preprocessor
        self.subject_ids = list(subject_ids)
        self.samples: List[Dict[str, Any]] = []
        id_col = preprocessor.config.id_col
        grouped = {rid: g for rid, g in df[df[id_col].isin(self.subject_ids)].groupby(id_col)}
        for rid in self.subject_ids:
            if rid not in grouped:
                continue
            try:
                sample = preprocessor.transform_group(grouped[rid])
                if sample["n_visits"] >= preprocessor.config.min_visits:
                    self.samples.append(sample)
            except Exception as e:
                print(f"[WARN] Dropping subject {rid}: {e}")
        if len(self.samples) == 0:
            raise ValueError("Dataset has no valid samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "rid": s["rid"],
            "x_num": torch.tensor(s["x_num"], dtype=torch.float32),
            "x_num_mask": torch.tensor(s["x_num_mask"], dtype=torch.float32),
            "x_cat": torch.tensor(s["x_cat"], dtype=torch.long),
            "times": torch.tensor(s["times"], dtype=torch.float32),
            "delta_t": torch.tensor(s["delta_t"], dtype=torch.float32),
            "duration": torch.tensor(s["duration"], dtype=torch.float32),
            "event": torch.tensor(s["event"], dtype=torch.float32),
        }


def collate_visit_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)
    L = max(item["x_num"].shape[0] for item in batch)
    Dn = batch[0]["x_num"].shape[1]
    Dcat = batch[0]["x_cat"].shape[1]

    x_num = torch.zeros(B, L, Dn, dtype=torch.float32)
    x_num_mask = torch.zeros(B, L, Dn, dtype=torch.float32)
    x_cat = torch.zeros(B, L, Dcat, dtype=torch.long)
    times = torch.zeros(B, L, dtype=torch.float32)
    delta_t = torch.zeros(B, L, dtype=torch.float32)
    visit_mask = torch.zeros(B, L, dtype=torch.bool)  # True for valid visits.
    durations = torch.zeros(B, dtype=torch.float32)
    events = torch.zeros(B, dtype=torch.float32)
    rids = []

    for i, item in enumerate(batch):
        l = item["x_num"].shape[0]
        x_num[i, :l] = item["x_num"]
        x_num_mask[i, :l] = item["x_num_mask"]
        x_cat[i, :l] = item["x_cat"]
        times[i, :l] = item["times"]
        delta_t[i, :l] = item["delta_t"]
        visit_mask[i, :l] = True
        durations[i] = item["duration"]
        events[i] = item["event"]
        rids.append(item["rid"])

    return {
        "rid": rids,
        "x_num": x_num,
        "x_num_mask": x_num_mask,
        "x_cat": x_cat,
        "times": times,
        "delta_t": delta_t,
        "visit_mask": visit_mask,
        "duration": durations,
        "event": events,
    }


# -----------------------------
# Model components
# -----------------------------

class VisitTokenizer(nn.Module):
    """Convert each tabular visit into a dense visit token."""

    def __init__(
        self,
        num_dim: int,
        cat_vocab_sizes: List[int],
        d_model: int = 128,
        cat_emb_dim: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_dim = num_dim
        self.cat_vocab_sizes = cat_vocab_sizes
        self.d_model = d_model

        self.num_mlp = nn.Sequential(
            nn.Linear(num_dim * 2, d_model),  # numeric values + numeric missingness mask
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, cat_emb_dim, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])
        cat_total_dim = cat_emb_dim * len(cat_vocab_sizes)
        self.cat_proj = nn.Linear(cat_total_dim, d_model) if cat_total_dim > 0 else None

        self.time_mlp = nn.Sequential(
            nn.Linear(2, d_model),  # absolute time and delta time
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_num: torch.Tensor,
        x_num_mask: torch.Tensor,
        x_cat: torch.Tensor,
        times: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        # x_num: [B, L, D], x_num_mask: [B, L, D], x_cat: [B, L, C]
        num_token = self.num_mlp(torch.cat([x_num, x_num_mask], dim=-1))
        if self.cat_proj is not None and x_cat.shape[-1] > 0:
            emb_list = []
            for j, emb in enumerate(self.cat_embeddings):
                emb_list.append(emb(x_cat[:, :, j]))
            cat_token = self.cat_proj(torch.cat(emb_list, dim=-1))
        else:
            cat_token = torch.zeros_like(num_token)
        time_input = torch.stack([times, delta_t], dim=-1)
        time_token = self.time_mlp(time_input)
        return self.out_norm(num_token + cat_token + time_token)


class ProgressionAwareMultiHorizonSurvivalModel(nn.Module):
    def __init__(self, num_dim: int, cat_vocab_sizes: List[int], n_bins: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_bins = n_bins
        self.visit_tokenizer = VisitTokenizer(
            num_dim=num_dim,
            cat_vocab_sizes=cat_vocab_sizes,
            d_model=cfg.d_model,
            cat_emb_dim=cfg.cat_emb_dim,
            dropout=cfg.dropout,
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_visits + 1, cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.visit_pool_score = nn.Linear(cfg.d_model, 1)
        self.representation_fuser = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # Progression score: higher means closer to conversion / faster progression.
        self.progression_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # Subtype-aware mixture survival head.
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts)
        self.expert_hazard_logits = nn.Linear(cfg.d_model, cfg.n_experts * n_bins)

    def _apply_visit_dropout(self, visit_mask: torch.Tensor) -> torch.Tensor:
        """Randomly drop observed visits during training while always keeping the latest visit."""
        if (not self.training) or self.cfg.visit_dropout <= 0:
            return visit_mask
        keep = torch.rand_like(visit_mask.float()) > float(self.cfg.visit_dropout)
        keep = keep & visit_mask

        # Always keep the most recent valid visit for each subject; it is usually the
        # clinically richest pre-index snapshot and prevents empty sequences.
        lengths = visit_mask.long().sum(dim=1).clamp_min(1)
        last_idx = (lengths - 1).view(-1, 1)
        keep.scatter_(1, last_idx, True)
        return keep & visit_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = self.visit_tokenizer(
            batch["x_num"], batch["x_num_mask"], batch["x_cat"], batch["times"], batch["delta_t"]
        )
        B, L, D = tokens.shape
        if L > self.cfg.max_visits:
            raise ValueError(f"Sequence length {L} > max_visits {self.cfg.max_visits}. Increase --max_visits.")
        effective_visit_mask = self._apply_visit_dropout(batch["visit_mask"])
        tokens = tokens * effective_visit_mask.unsqueeze(-1).to(tokens.dtype)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_emb[:, : L + 1, :]

        # Transformer expects True for padded positions. CLS is never padded.
        pad_mask = ~effective_visit_mask
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
        key_padding_mask = torch.cat([cls_pad, pad_mask], dim=1)

        z_all = self.encoder(x, src_key_padding_mask=key_padding_mask)
        z_all = self.encoder_norm(z_all)
        z_cls = z_all[:, 0, :]
        z_visits = z_all[:, 1:, :]

        pool_logits = self.visit_pool_score(z_visits).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~effective_visit_mask, -1e9)
        pool_weights = torch.softmax(pool_logits, dim=1)
        z_pool = (pool_weights.unsqueeze(-1) * z_visits).sum(dim=1)

        visit_positions = torch.arange(L, device=effective_visit_mask.device).view(1, L).expand(B, -1)
        last_idx = visit_positions.masked_fill(~effective_visit_mask, -1).max(dim=1).values.clamp_min(0)
        last_idx = last_idx.view(B, 1, 1).expand(-1, 1, D)
        z_last = z_visits.gather(dim=1, index=last_idx).squeeze(1)
        z = self.representation_fuser(torch.cat([z_cls, z_pool, z_last], dim=-1))

        progression = self.progression_head(z).squeeze(-1)
        gate_weights = F.softmax(self.gate(z), dim=-1)  # [B, E]
        expert_logits = self.expert_hazard_logits(z).view(B, self.cfg.n_experts, self.n_bins)
        expert_hazards = torch.sigmoid(expert_logits).clamp(1e-6, 1.0 - 1e-6)
        hazards = (gate_weights.unsqueeze(-1) * expert_hazards).sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = 1.0 - survival
        return {
            "visit_tokens": tokens,
            "z": z,
            "pool_weights": pool_weights,
            "progression": progression,
            "gate_weights": gate_weights,
            "hazards": hazards,
            "survival": survival,
            "risk": risk,
        }


# -----------------------------
# Losses and metrics
# -----------------------------

def discrete_survival_nll(
    hazards: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    bins: torch.Tensor,
    event_weight: float = 1.0,
) -> torch.Tensor:
    """
    Discrete-time survival negative log likelihood.

    hazards[:, k] = P(event in (bins[k-1], bins[k]] | survived before bin k).
    For an event in bin k: log survival through previous bins + log hazard in event bin.
    For censored at time c: log survival through fully observed bins with bin endpoint <= c.
    If event occurs after the largest modeled bin, it contributes survival through all bins.
    """
    eps = 1e-7
    hazards = hazards.clamp(eps, 1.0 - eps)
    log_h = torch.log(hazards)
    log_surv_step = torch.log1p(-hazards)
    B, K = hazards.shape
    loss = torch.zeros(B, device=hazards.device)

    event_bin = torch.bucketize(durations, bins, right=False)  # 0..K
    censor_count = torch.bucketize(durations, bins, right=True).clamp(max=K)  # number of full bins observed

    for i in range(B):
        if events[i] > 0.5 and event_bin[i] < K:
            k = int(event_bin[i].item())
            if k > 0:
                loss[i] = loss[i] - log_surv_step[i, :k].sum()
            loss[i] = loss[i] - log_h[i, k]
        else:
            c = int(censor_count[i].item())
            if c > 0:
                loss[i] = loss[i] - log_surv_step[i, :c].sum()
        if events[i] > 0.5:
            loss[i] = loss[i] * event_weight
    return loss.mean()


def horizon_risk_from_hazards(hazards: torch.Tensor, bins: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
    survival = torch.cumprod(1.0 - hazards.clamp(1e-7, 1.0 - 1e-7), dim=1)
    risks = []
    for h in horizons:
        idx = torch.searchsorted(bins, h, right=False)
        idx = torch.clamp(idx, max=bins.numel() - 1)
        risks.append(1.0 - survival[:, idx])
    return torch.stack(risks, dim=1)  # [B, H]


def horizon_bce_loss(
    hazards: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    bins: torch.Tensor,
    horizons: torch.Tensor,
    pos_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    risks = horizon_risk_from_hazards(hazards, bins, horizons).clamp(1e-6, 1.0 - 1e-6)
    labels = []
    masks = []
    for h in horizons:
        y = ((events > 0.5) & (durations <= h)).float()
        evaluable = (((events > 0.5) & (durations <= h)) | (durations >= h)).float()
        labels.append(y)
        masks.append(evaluable)
    y = torch.stack(labels, dim=1)
    mask = torch.stack(masks, dim=1)
    bce = F.binary_cross_entropy(risks, y, reduction="none")
    if focal_gamma > 0:
        p_t = torch.where(y > 0.5, risks, 1.0 - risks).clamp(1e-6, 1.0)
        bce = bce * torch.pow(1.0 - p_t, float(focal_gamma))
    if pos_weights is not None and pos_weights.numel() == y.shape[1]:
        weights = torch.where(y > 0.5, pos_weights.view(1, -1), torch.ones_like(y))
    else:
        weights = torch.ones_like(y)
    weights = weights * mask
    denom = weights.sum().clamp_min(1.0)
    return (bce * weights).sum() / denom


def hazard_smoothness_loss(hazards: torch.Tensor) -> torch.Tensor:
    """Small adjacent-bin smoothness penalty to reduce overfit survival curves."""
    if hazards.shape[1] < 2:
        return hazards.sum() * 0.0
    return torch.square(hazards[:, 1:] - hazards[:, :-1]).mean()


def gate_balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Manuscript Eq. (10): sum_e (mean_i(pi_ie) - 1/E)**2.

    Average each expert's gate weight over the minibatch, then sum the squared
    deviations across experts. There is no additional division by E.
    """
    if gate_weights.numel() == 0 or gate_weights.shape[1] <= 1:
        return gate_weights.sum() * 0.0
    mean_gate = gate_weights.mean(dim=0)
    target = torch.full_like(mean_gate, 1.0 / gate_weights.shape[1])
    return torch.square(mean_gate - target).sum()


def progression_ranking_loss(progression: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking: subjects with earlier observed events should have higher progression scores.
    Comparable pair: i has event and T_i < T_j.
    """
    p_i = progression.view(-1, 1)
    p_j = progression.view(1, -1)
    t_i = durations.view(-1, 1)
    t_j = durations.view(1, -1)
    e_i = events.view(-1, 1) > 0.5
    comparable = e_i & (t_i < t_j)
    if comparable.sum() == 0:
        return progression.sum() * 0.0
    # Want p_i > p_j.
    margin_loss = F.softplus(-(p_i - p_j))
    return margin_loss[comparable].mean()


def total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    bins: torch.Tensor,
    horizons: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    surv = discrete_survival_nll(outputs["hazards"], batch["duration"], batch["event"], bins, cfg.event_weight)
    pos_weights = None
    if cfg.horizon_pos_weights:
        pos_weights = torch.tensor(cfg.horizon_pos_weights, dtype=outputs["hazards"].dtype, device=outputs["hazards"].device)
    hbce = horizon_bce_loss(
        outputs["hazards"],
        batch["duration"],
        batch["event"],
        bins,
        horizons,
        pos_weights=pos_weights,
        focal_gamma=cfg.horizon_focal_gamma,
    )
    prank = progression_ranking_loss(outputs["progression"], batch["duration"], batch["event"])
    smooth = hazard_smoothness_loss(outputs["hazards"])
    gate_bal = gate_balance_loss(outputs["gate_weights"])
    loss = (
        surv
        + cfg.lambda_horizon * hbce
        + cfg.lambda_progression * prank
        + cfg.lambda_smooth * smooth
        + cfg.lambda_gate_balance * gate_bal
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "surv_nll": float(surv.detach().cpu()),
        "horizon_bce": float(hbce.detach().cpu()),
        "progression_rank": float(prank.detach().cpu()),
        "hazard_smooth": float(smooth.detach().cpu()),
        "gate_balance": float(gate_bal.detach().cpu()),
    }


def concordance_index_simple(durations: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    concordant = 0.0
    permissible = 0.0
    n = len(durations)
    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if durations[i] < durations[j]:
                permissible += 1.0
                if risks[i] > risks[j]:
                    concordant += 1.0
                elif risks[i] == risks[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible > 0 else float("nan")


def best_threshold_by_balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if not SKLEARN_AVAILABLE or len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.05, 0.95, 91)))
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def safe_metric(fn, *args, default=float("nan"), **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {"n": int(len(y_true)), "threshold": float(threshold)}
    if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
        out.update({
            "auroc": safe_metric(roc_auc_score, y_true, y_prob),
            "auprc": safe_metric(average_precision_score, y_true, y_prob),
            "brier": safe_metric(brier_score_loss, y_true, y_prob),
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
            "accuracy": safe_metric(accuracy_score, y_true, y_pred),
            "precision": safe_metric(precision_score, y_true, y_pred, zero_division=0),
            "recall_sensitivity": safe_metric(recall_score, y_true, y_pred, zero_division=0),
            "f1": safe_metric(f1_score, y_true, y_pred, zero_division=0),
        })
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            out.update({
                "specificity": float(tn / max(tn + fp, 1)),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            })
        except Exception:
            pass
    else:
        out.update({
            "auroc": float("nan"), "auprc": float("nan"), "brier": float("nan"),
            "balanced_accuracy": float("nan"), "accuracy": float(np.mean(y_pred == y_true)) if len(y_true) else float("nan"),
        })
    return out


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, bins: torch.Tensor, horizons: torch.Tensor) -> pd.DataFrame:
    model.eval()
    rows = []
    bins = bins.to(device)
    horizons = horizons.to(device)
    for batch in loader:
        for k in ["x_num", "x_num_mask", "x_cat", "times", "delta_t", "visit_mask", "duration", "event"]:
            batch[k] = batch[k].to(device)
        out = model(batch)
        hrisks = horizon_risk_from_hazards(out["hazards"], bins, horizons)
        risk_max = out["risk"][:, -1]
        hazards = out["hazards"].detach().cpu().numpy()
        for i, rid in enumerate(batch["rid"]):
            row = {
                "RID": rid,
                "duration": float(batch["duration"][i].detach().cpu()),
                "event": int(batch["event"][i].detach().cpu().item()),
                "risk_max_bin": float(risk_max[i].detach().cpu()),
                "progression_score": float(out["progression"][i].detach().cpu()),
            }
            for j, h in enumerate(horizons.detach().cpu().numpy().tolist()):
                row[f"risk_{h:g}y"] = float(hrisks[i, j].detach().cpu())
            for j in range(hazards.shape[1]):
                row[f"hazard_bin{j+1}"] = float(hazards[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(pred: pd.DataFrame, horizons: List[float], val_thresholds: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, Any], Dict[str, float]]:
    durations = pred["duration"].to_numpy(float)
    events = pred["event"].to_numpy(int)
    risk_max = pred["risk_max_bin"].to_numpy(float)
    metrics: Dict[str, Any] = {
        "n_subjects": int(len(pred)),
        "n_events": int(events.sum()),
        "event_rate": float(events.mean()) if len(events) else float("nan"),
        "c_index_risk_max_bin": concordance_index_simple(durations, events, risk_max),
    }
    thresholds: Dict[str, float] = {}
    horizon_metrics = {}
    for h in horizons:
        risk_col = f"risk_{h:g}y"
        y = ((events == 1) & (durations <= h)).astype(int)
        evaluable = (((events == 1) & (durations <= h)) | (durations >= h))
        y_eval = y[evaluable]
        p_eval = pred.loc[evaluable, risk_col].to_numpy(float)
        key = f"{h:g}y"
        if val_thresholds and key in val_thresholds:
            t = val_thresholds[key]
        else:
            t = best_threshold_by_balanced_accuracy(y_eval, p_eval)
        thresholds[key] = float(t)
        horizon_metrics[key] = binary_metrics(y_eval, p_eval, threshold=t)
        horizon_metrics[key]["n_evaluable"] = int(evaluable.sum())
        horizon_metrics[key]["n_positive"] = int(y_eval.sum())
    metrics["horizon_metrics"] = horizon_metrics
    return metrics, thresholds


# -----------------------------
# Training utilities
# -----------------------------

def subject_level_split(
    df: pd.DataFrame,
    id_col: str,
    event_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
    sub = df.groupby(id_col)[event_col].first().reset_index()
    ids = sub[id_col].to_numpy()
    y = sub[event_col].to_numpy()
    stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 2 else None
    if SKLEARN_AVAILABLE:
        trainval_ids, test_ids, y_trainval, _ = train_test_split(
            ids, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        stratify2 = y_trainval if len(np.unique(y_trainval)) == 2 and min(np.bincount(y_trainval.astype(int))) >= 2 else None
        val_relative = val_size / (1.0 - test_size)
        train_ids, val_ids = train_test_split(
            trainval_ids, test_size=val_relative, random_state=seed, stratify=stratify2
        )
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ids)
        n_test = int(round(len(perm) * test_size))
        n_val = int(round(len(perm) * val_size))
        test_ids = perm[:n_test]
        val_ids = perm[n_test:n_test+n_val]
        train_ids = perm[n_test+n_val:]
    return list(train_ids), list(val_ids), list(test_ids)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    bins: torch.Tensor,
    horizons: torch.Tensor,
    cfg: TrainConfig,
    ema: Optional["ModelEMA"] = None,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: Dict[str, float] = {
        "loss": 0.0,
        "surv_nll": 0.0,
        "horizon_bce": 0.0,
        "progression_rank": 0.0,
        "hazard_smooth": 0.0,
        "gate_balance": 0.0,
    }
    n_batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        bins_dev = bins.to(device)
        horizons_dev = horizons.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, parts = total_loss(outputs, batch, bins_dev, horizons_dev, cfg)
        if train_mode:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        for k, v in parts.items():
            totals[k] += float(v)
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


class ModelEMA:
    """Exponential moving average of trainable weights for stabler validation checkpoints."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


def make_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.lr_scheduler == "none":
        return None
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_plateau_factor,
            patience=cfg.lr_plateau_patience,
            min_lr=cfg.min_lr,
        )
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(cfg.epochs), 1),
            eta_min=cfg.min_lr,
        )
    raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")


def validation_score_from_parts(parts: Dict[str, float], metric_name: str) -> float:
    key_map = {
        "val_loss": "loss",
        "val_surv_nll": "surv_nll",
        "val_horizon_bce": "horizon_bce",
    }
    if metric_name not in key_map:
        raise ValueError(f"Unsupported selection_metric: {metric_name}")
    return float(parts[key_map[metric_name]])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    bins: torch.Tensor,
    horizons: torch.Tensor,
    cfg: TrainConfig,
    out_dir: str,
) -> str:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_lr_scheduler(optimizer, cfg)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None
    best_val = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")
    no_improve = 0

    history = []
    for epoch in range(1, cfg.epochs + 1):
        tr = run_one_epoch(model, train_loader, optimizer, device, bins, horizons, cfg, ema=ema)
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        va = run_one_epoch(model, val_loader, None, device, bins, horizons, cfg)
        score = validation_score_from_parts(va, cfg.selection_metric)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        row["selection_metric"] = cfg.selection_metric
        row["selection_score"] = score
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} "
            f"| val survival {va['surv_nll']:.4f} | val horizon {va['horizon_bce']:.4f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if scheduler is not None:
            if cfg.lr_scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()
        if score < best_val - cfg.min_delta:
            best_val = score
            no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": va["loss"],
                    "selection_metric": cfg.selection_metric,
                    "selection_score": best_val,
                    "ema_decay": cfg.ema_decay if ema is not None else 0.0,
                },
                best_path,
            )
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best {cfg.selection_metric}={best_val:.4f}")
                if ema is not None:
                    ema.restore(model)
                break
        if ema is not None:
            ema.restore(model)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    return best_path


@torch.no_grad()
def export_dense_visit_tokens(model: ProgressionAwareMultiHorizonSurvivalModel, loader: DataLoader, device: torch.device, out_path: str) -> None:
    """Save learned dense visit tokens after training. Variable-length tokens are stored as object arrays."""
    model.eval()
    rids, token_list, time_list = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        tokens = model.visit_tokenizer(batch["x_num"], batch["x_num_mask"], batch["x_cat"], batch["times"], batch["delta_t"])
        for i, rid in enumerate(batch["rid"]):
            l = int(batch["visit_mask"][i].sum().detach().cpu())
            rids.append(rid)
            token_list.append(tokens[i, :l].detach().cpu().numpy())
            time_list.append(batch["times"][i, :l].detach().cpu().numpy())
    np.savez_compressed(
        out_path,
        RID=np.asarray(rids, dtype=object),
        visit_tokens=np.asarray(token_list, dtype=object),
        visit_times=np.asarray(time_list, dtype=object),
    )


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Progression-aware multihorizon survival model for AD progression. "
            "Supports either one unsplit CSV or three pre-split CSV files."
        )
    )

    # Data input. Keep --csv_path for backward compatibility; use the three paths
    # below when train/validation/test splits have already been created.
    p.add_argument("--csv_path", type=str, default=None,
                   help="Optional single unsplit CSV. Used only when pre-split CSV paths are not provided.")
    p.add_argument("--train_csv_path", type=str, default=None,
                   help="Training CSV containing longitudinal pre-index visit rows.")
    p.add_argument("--val_csv_path", type=str, default=None,
                   help="Validation CSV containing longitudinal pre-index visit rows.")
    p.add_argument("--test_csv_path", type=str, default=None,
                   help="Testing CSV containing longitudinal pre-index visit rows.")
    p.add_argument("--out_dir", type=str, required=True)

    # Column names.
    p.add_argument("--id_col", type=str, default="RID")
    p.add_argument("--date_col", type=str, default="EXAMDATE")
    p.add_argument("--time_col", type=str, default="time_interval_to_first_visit_years")
    p.add_argument("--duration_col", type=str, default="time_to_index_years")
    p.add_argument("--event_col", type=str, default="label")
    p.add_argument("--index_date_col", type=str, default="index_date")
    p.add_argument("--categorical_cols", nargs="*", default=["PTGENDER", "APOE4"],
                   help="Leakage-safe categorical predictors. Accepts space- or comma-separated values. Default: PTGENDER APOE4.")
    p.add_argument("--drop_cols", nargs="*",
                   default=["DX", "DXCHANGE", "baseline_stage", "visit_stage", "diagnosis", "Diagnosis"],
                   help="Additional input columns to exclude before feature inference. Diagnosis/stage fields are dropped by default.")
    p.add_argument("--calibration_method", type=str, default="isotonic", choices=["none", "isotonic", "platt"],
                   help="Validation-set calibration applied to horizon risks for reported horizon metrics.")
    p.add_argument("--threshold_strategy", type=str, default="youden", choices=["youden", "f1", "fixed_0.5"],
                   help="Validation-set threshold strategy for horizon classification metrics in standardized CSV outputs.")
    p.add_argument("--n_eval_grid", type=int, default=50, help="Evaluation grid size for time-dependent Brier/IBS when scikit-survival is installed.")
    p.add_argument("--calibration_bins", type=int, default=5, help="Number of quantile bins for calibration tables.")
    p.add_argument("--no_strict_preindex_filter", action="store_true")
    p.add_argument(
        "--disable_time_derived_features",
        action="store_true",
        help=(
            "Timing-sensitivity condition: zero the absolute/inter-visit time token "
            "and remove time-normalized slopes while retaining visit order and "
            "delta-from-first feature values."
        ),
    )

    # Sequence settings.
    p.add_argument("--min_visits", type=int, default=1)
    p.add_argument("--max_visits", type=int, default=32)

    # Used only for the single unsplit CSV mode.
    p.add_argument("--test_size", type=float, default=0.20,
                   help="Test fraction used only with --csv_path.")
    p.add_argument("--val_size", type=float, default=0.20,
                   help="Validation fraction used only with --csv_path.")

    # Survival/horizon settings.
    p.add_argument("--horizons", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    p.add_argument("--bins", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

    # Training settings.
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--event_weight", type=float, default=2.0)
    p.add_argument("--auto_event_weight", action="store_true",
                   help="Replace --event_weight with a clipped train-set inverse event-frequency weight.")
    p.add_argument("--max_event_weight", type=float, default=5.0)
    p.add_argument("--lambda_horizon", type=float, default=0.25)
    p.add_argument("--lambda_progression", type=float, default=0.10)
    p.add_argument("--lambda_smooth", type=float, default=0.02,
                   help="Adjacent hazard smoothness penalty. Useful for small noisy ADNI splits.")
    p.add_argument("--lambda_gate_balance", type=float, default=0.01,
                   help="Batch-level expert gate balance penalty for the subtype mixture head.")
    p.add_argument("--horizon_focal_gamma", type=float, default=1.0,
                   help="Focal exponent for horizon BCE. Set 0 to recover ordinary BCE.")
    p.add_argument("--auto_horizon_pos_weights", action="store_true",
                   help="Use clipped inverse-frequency positive weights for each reported horizon.")
    p.add_argument("--max_horizon_pos_weight", type=float, default=8.0)
    p.add_argument("--selection_metric", type=str, default="val_loss",
                   choices=["val_loss", "val_surv_nll", "val_horizon_bce"],
                   help="Validation objective used for checkpoint selection and early stopping.")
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)
    p.add_argument("--lr_plateau_patience", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--ema_decay", type=float, default=0.995,
                   help="EMA decay for checkpoint evaluation. Set 0 to disable.")

    # Model settings.
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--cat_emb_dim", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--n_experts", type=int, default=3)
    p.add_argument("--visit_dropout", type=float, default=0.10,
                   help="Randomly drop observed visits during training while keeping each subject's latest visit.")

    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--export_visit_tokens", action="store_true",
                   help="Save learned dense visit tokens for all splits after training.")

    args = p.parse_args()

    has_any_presplit = any([args.train_csv_path, args.val_csv_path, args.test_csv_path])
    has_all_presplit = all([args.train_csv_path, args.val_csv_path, args.test_csv_path])
    if has_any_presplit and not has_all_presplit:
        raise ValueError(
            "When using pre-split files, please provide all three: "
            "--train_csv_path, --val_csv_path, and --test_csv_path."
        )
    if not has_any_presplit and args.csv_path is None:
        raise ValueError(
            "Please provide either --csv_path for a single unsplit CSV or "
            "--train_csv_path, --val_csv_path, and --test_csv_path for pre-split CSVs."
        )
    return args


def load_longitudinal_csv(path: str, split_name: str, args: argparse.Namespace) -> pd.DataFrame:
    """Read one split CSV and apply light schema checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name} CSV not found: {path}")
    df = pd.read_csv(path)

    required = [args.id_col, args.duration_col, args.event_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} CSV is missing required columns: {missing}")

    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    if args.date_col not in df.columns and args.time_col not in df.columns:
        print(
            f"[WARN] {split_name} CSV has neither {args.date_col} nor {args.time_col}. "
            "Visits will be processed in the input row order."
        )

    # Keep event and duration numeric. Rows with missing subject id are not useful.
    df = df[df[args.id_col].notna()].copy()
    df[args.event_col] = pd.to_numeric(df[args.event_col], errors="coerce")
    df[args.duration_col] = pd.to_numeric(df[args.duration_col], errors="coerce")

    before = len(df)
    df = df[df[args.event_col].notna() & df[args.duration_col].notna()].copy()
    if len(df) < before:
        print(f"[WARN] Dropped {before - len(df)} rows from {split_name} with missing event/duration.")

    return df


def get_subject_ids(df: pd.DataFrame, id_col: str) -> List[Any]:
    """Return unique subject IDs preserving first-seen order."""
    return df[id_col].drop_duplicates().tolist()


def check_no_subject_overlap(train_ids: List[Any], val_ids: List[Any], test_ids: List[Any]) -> None:
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set
    if tv or tt or vt:
        msg = []
        if tv:
            msg.append(f"train/val overlap={len(tv)}")
        if tt:
            msg.append(f"train/test overlap={len(tt)}")
        if vt:
            msg.append(f"val/test overlap={len(vt)}")
        example = list(tv or tt or vt)[:10]
        raise ValueError(
            "Subject-level split violation detected: "
            + ", ".join(msg)
            + f". Example overlapping IDs: {example}"
        )


def warn_if_subject_targets_vary(df: pd.DataFrame, split_name: str, id_col: str, duration_col: str, event_col: str) -> None:
    """Warn when a subject has inconsistent event/duration values across visit rows."""
    bad = []
    for rid, g in df.groupby(id_col):
        dur_unique = pd.to_numeric(g[duration_col], errors="coerce").dropna().unique()
        ev_unique = pd.to_numeric(g[event_col], errors="coerce").dropna().unique()
        if len(dur_unique) > 1 or len(ev_unique) > 1:
            bad.append(rid)
    if bad:
        print(
            f"[WARN] {split_name}: {len(bad)} subjects have inconsistent "
            f"{duration_col} or {event_col} across visit rows. "
            f"The first non-missing value is used. Examples: {bad[:10]}"
        )


def align_feature_columns(df: pd.DataFrame, pre: LongitudinalVisitPreprocessor, split_name: str) -> pd.DataFrame:
    """
    Ensure validation/test CSVs contain the feature columns inferred from the training CSV.
    Missing columns are added as NaN so they become imputed/masked rather than causing failure.
    """
    df = df.copy()
    added = []
    for col in pre.numeric_cols + pre.categorical_cols:
        if col not in df.columns:
            df[col] = np.nan
            added.append(col)
    if added:
        print(f"[WARN] Added {len(added)} missing train-fitted feature columns to {split_name}: {added[:12]}")
    return df


def make_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pieces = []
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sub = (
            df.groupby(args.id_col)
            .agg(
                n_visit_rows=(args.id_col, "size"),
                event=(args.event_col, "first"),
                duration=(args.duration_col, "first"),
            )
            .reset_index()
        )
        sub["split"] = split_name
        pieces.append(sub)
    return pd.concat(pieces, axis=0, ignore_index=True)


def print_split_summary(split_df: pd.DataFrame, id_col: str) -> None:
    print("\nSubject-level split summary:")
    for split_name, g in split_df.groupby("split", sort=False):
        n = len(g)
        n_events = int(pd.to_numeric(g["event"], errors="coerce").fillna(0).sum())
        mean_visits = float(pd.to_numeric(g["n_visit_rows"], errors="coerce").mean())
        print(f"  {split_name:5s}: subjects={n:4d}, events={n_events:4d}, event_rate={n_events/max(n,1):.3f}, mean_visits={mean_visits:.2f}")


def clipped_inverse_event_weight(events: np.ndarray, max_weight: float) -> float:
    events = np.asarray(events, dtype=int)
    n_pos = int((events == 1).sum())
    n_neg = int((events == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(np.clip(n_neg / max(n_pos, 1), 1.0, max_weight))


def compute_horizon_pos_weights(
    samples: List[Dict[str, Any]],
    horizons: List[float],
    max_weight: float,
) -> Tuple[float, ...]:
    durations = np.asarray([float(s["duration"]) for s in samples], dtype=float)
    events = np.asarray([int(s["event"]) for s in samples], dtype=int)
    weights = []
    for h in horizons:
        positive = (events == 1) & (durations <= float(h))
        evaluable = positive | (durations >= float(h))
        n_pos = int(positive[evaluable].sum())
        n_neg = int(evaluable.sum() - n_pos)
        if n_pos <= 0 or n_neg <= 0:
            weights.append(1.0)
        else:
            weights.append(float(np.clip(n_neg / max(n_pos, 1), 1.0, max_weight)))
    return tuple(weights)


def prepare_dataframes(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Any], List[Any], List[Any], str]:
    """
    Return train/val/test dataframes and subject IDs.

    Mode 1: pre-split mode, when --train_csv_path/--val_csv_path/--test_csv_path are supplied.
    Mode 2: backward-compatible single-CSV mode, when --csv_path is supplied.
    """
    if args.train_csv_path and args.val_csv_path and args.test_csv_path:
        mode = "presplit"
        print("Using user-provided train/validation/test CSV files. No internal split will be created.")
        train_df = load_longitudinal_csv(args.train_csv_path, "train", args)
        val_df = load_longitudinal_csv(args.val_csv_path, "val", args)
        test_df = load_longitudinal_csv(args.test_csv_path, "test", args)

        train_ids = get_subject_ids(train_df, args.id_col)
        val_ids = get_subject_ids(val_df, args.id_col)
        test_ids = get_subject_ids(test_df, args.id_col)
        check_no_subject_overlap(train_ids, val_ids, test_ids)

    else:
        mode = "single_csv_internal_split"
        print("Using one unsplit CSV. Creating subject-level train/validation/test splits internally.")
        df = load_longitudinal_csv(args.csv_path, "all", args)
        train_ids, val_ids, test_ids = subject_level_split(
            df, args.id_col, args.event_col, args.test_size, args.val_size, args.seed
        )
        train_df = df[df[args.id_col].isin(train_ids)].copy()
        val_df = df[df[args.id_col].isin(val_ids)].copy()
        test_df = df[df[args.id_col].isin(test_ids)].copy()
        check_no_subject_overlap(train_ids, val_ids, test_ids)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        warn_if_subject_targets_vary(split_df, split_name, args.id_col, args.duration_col, args.event_col)

    return train_df, val_df, test_df, train_ids, val_ids, test_ids, mode


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    train_df, val_df, test_df, train_ids, val_ids, test_ids, input_mode = prepare_dataframes(args)

    categorical_cols = normalize_list_arg(args.categorical_cols)
    drop_cols = normalize_list_arg(args.drop_cols)
    data_cfg = DataConfig(
        id_col=args.id_col,
        date_col=args.date_col,
        time_col=args.time_col,
        duration_col=args.duration_col,
        event_col=args.event_col,
        index_date_col=args.index_date_col,
        min_visits=args.min_visits,
        max_visits=args.max_visits,
        strict_preindex_filter=not args.no_strict_preindex_filter,
        use_time_derived_features=not args.disable_time_derived_features,
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
    )

    # Fit preprocessing only on the training split. Validation/test use the same scaler,
    # imputer, categorical vocabularies, and feature columns.
    pre = LongitudinalVisitPreprocessor(data_cfg)
    pre.fit(train_df)
    pre.save(os.path.join(args.out_dir, "visit_preprocessor.pkl"))

    train_df = align_feature_columns(train_df, pre, "train")
    val_df = align_feature_columns(val_df, pre, "val")
    test_df = align_feature_columns(test_df, pre, "test")

    split_summary = make_split_summary(train_df, val_df, test_df, args)
    split_summary.to_csv(os.path.join(args.out_dir, "subject_splits.csv"), index=False)
    print_split_summary(split_summary, args.id_col)

    print(f"\nInput mode: {input_mode}")
    print(f"Numeric raw feature columns inferred from training set: {len(pre.numeric_cols)}")
    print(f"Categorical feature columns inferred from training set: {len(pre.categorical_cols)} -> {pre.categorical_cols}")

    train_ds = ADProgressionDataset(train_df, train_ids, pre)
    val_ds = ADProgressionDataset(val_df, val_ids, pre)
    test_ds = ADProgressionDataset(test_df, test_ids, pre)
    train_events_for_weight = np.asarray([int(s["event"]) for s in train_ds.samples], dtype=int)
    effective_event_weight = float(args.event_weight)
    if args.auto_event_weight:
        effective_event_weight = clipped_inverse_event_weight(train_events_for_weight, args.max_event_weight)
    horizon_pos_weights: Tuple[float, ...] = ()
    if args.auto_horizon_pos_weights:
        horizon_pos_weights = compute_horizon_pos_weights(
            train_ds.samples,
            [float(x) for x in args.horizons],
            args.max_horizon_pos_weight,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_visit_sequences,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_visit_sequences,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_visit_sequences,
        num_workers=args.num_workers,
    )

    sample = train_ds[0]
    num_dim = sample["x_num"].shape[1]
    cat_vocab_sizes = [len(pre.vocabs[c]) for c in pre.categorical_cols]
    bins = torch.tensor(sorted(args.bins), dtype=torch.float32)
    horizons = torch.tensor(args.horizons, dtype=torch.float32)

    model_cfg = ModelConfig(
        d_model=args.d_model,
        cat_emb_dim=args.cat_emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_experts=args.n_experts,
        max_visits=args.max_visits,
        visit_dropout=args.visit_dropout,
    )
    train_cfg = TrainConfig(
        horizons=tuple(args.horizons),
        bins=tuple(sorted(args.bins)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        event_weight=effective_event_weight,
        lambda_horizon=args.lambda_horizon,
        lambda_progression=args.lambda_progression,
        lambda_smooth=args.lambda_smooth,
        lambda_gate_balance=args.lambda_gate_balance,
        horizon_pos_weights=horizon_pos_weights,
        horizon_focal_gamma=args.horizon_focal_gamma,
        patience=args.patience,
        min_delta=args.min_delta,
        selection_metric=args.selection_metric,
        lr_scheduler=args.lr_scheduler,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_patience=args.lr_plateau_patience,
        min_lr=args.min_lr,
        ema_decay=args.ema_decay,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    input_paths = {
        "input_mode": input_mode,
        "csv_path": args.csv_path,
        "train_csv_path": args.train_csv_path,
        "val_csv_path": args.val_csv_path,
        "test_csv_path": args.test_csv_path,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "input_paths": input_paths,
                "data": asdict(data_cfg),
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "standardized_evaluation": {
                    "horizons": [float(x) for x in args.horizons],
                    "overall_risk_horizon_years": float(max(args.horizons)),
                    "threshold_strategy": args.threshold_strategy,
                    "calibration_method": args.calibration_method,
                    "n_eval_grid": int(args.n_eval_grid),
                    "calibration_bins": int(args.calibration_bins),
                },
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump(
            {
                "numeric_cols": pre.numeric_cols,
                "categorical_cols": pre.categorical_cols,
                "cat_vocab_sizes": cat_vocab_sizes,
                "num_dim_after_engineering": int(num_dim),
            },
            f,
            indent=2,
        )

    device = torch.device(args.device)
    model = ProgressionAwareMultiHorizonSurvivalModel(
        num_dim=num_dim,
        cat_vocab_sizes=cat_vocab_sizes,
        n_bins=len(bins),
        cfg=model_cfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nDataset after preprocessing:")
    print(f"  train subjects={len(train_ds)}, val subjects={len(val_ds)}, test subjects={len(test_ds)}")
    print(f"  numeric feature dimension after value/delta/slope engineering={num_dim}")
    print(f"  categorical vocab sizes={cat_vocab_sizes}")
    print(f"  modeled survival bins={sorted(args.bins)}")
    print(f"  reported horizons={args.horizons}")
    print(f"  effective event weight={effective_event_weight:.3f}")
    if horizon_pos_weights:
        print(f"  horizon positive weights={list(horizon_pos_weights)}")
    print(f"Model trainable parameters: {n_params:,}")

    best_path = train_model(model, train_loader, val_loader, device, bins, horizons, train_cfg, args.out_dir)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(
        f"Loaded best model from epoch {ckpt['epoch']} with val loss {ckpt.get('val_loss', float('nan')):.4f} "
        f"and {ckpt.get('selection_metric', 'selection_score')}={ckpt.get('selection_score', float('nan')):.4f}"
    )

    pred_train = predict(model, train_loader, device, bins, horizons)
    pred_train.to_csv(os.path.join(args.out_dir, "predictions_train.csv"), index=False)

    pred_val = predict(model, val_loader, device, bins, horizons)
    val_metrics, val_thresholds = evaluate_predictions(pred_val, args.horizons, val_thresholds=None)
    pred_val.to_csv(os.path.join(args.out_dir, "predictions_val.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics_val.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "val_thresholds.json"), "w") as f:
        json.dump(val_thresholds, f, indent=2)

    pred_test = predict(model, test_loader, device, bins, horizons)
    test_metrics, _ = evaluate_predictions(pred_test, args.horizons, val_thresholds=val_thresholds)
    pred_test.to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Standardized survival/horizon evaluation shared with baseline scripts.
    # This creates CSV files that can be aggregated across seeds and compared
    # directly with classical survival, sequence survival, and modern baselines.
    try:
        std_tables = standard_evaluate_survival_predictions(
            model_name="promise_ad",
            train_events=pred_train["event"].to_numpy(int),
            train_durations=pred_train["duration"].to_numpy(float),
            val_events=pred_val["event"].to_numpy(int),
            val_durations=pred_val["duration"].to_numpy(float),
            val_hazards=hazards_from_prediction_dataframe(pred_val),
            test_events=pred_test["event"].to_numpy(int),
            test_durations=pred_test["duration"].to_numpy(float),
            test_hazards=hazards_from_prediction_dataframe(pred_test),
            bin_edges_or_endpoints=np.asarray(sorted(args.bins), dtype=float),
            horizons=args.horizons,
            overall_risk_horizon=float(max(args.horizons)),
            threshold_strategy=args.threshold_strategy,
            calibration_method=args.calibration_method,
            n_eval_grid=args.n_eval_grid,
            calibration_bins=args.calibration_bins,
            seed=args.seed,
        )
        std_tables["summary"].to_csv(os.path.join(args.out_dir, "proposed_survival_summary_metrics.csv"), index=False)
        std_tables["horizon"].to_csv(os.path.join(args.out_dir, "proposed_survival_horizon_metrics.csv"), index=False)
        std_tables["horizon"].to_csv(os.path.join(args.out_dir, "comparison_ready_proposed_survival_horizon_metrics.csv"), index=False)
        std_tables["time_dependent_auc"].to_csv(os.path.join(args.out_dir, "proposed_survival_time_dependent_auc.csv"), index=False)
        std_tables["brier_scores"].to_csv(os.path.join(args.out_dir, "proposed_survival_brier_scores.csv"), index=False)
        std_tables["calibration"].to_csv(os.path.join(args.out_dir, "proposed_survival_calibration_by_horizon.csv"), index=False)
        std_tables["km"].to_csv(os.path.join(args.out_dir, "proposed_survival_km_stratification.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Standardized survival evaluation failed: {e}")

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics using validation-selected thresholds:")
    print(json.dumps(test_metrics, indent=2))

    if args.export_visit_tokens:
        export_dense_visit_tokens(model, train_loader, device, os.path.join(args.out_dir, "visit_tokens_train.npz"))
        export_dense_visit_tokens(model, val_loader, device, os.path.join(args.out_dir, "visit_tokens_val.npz"))
        export_dense_visit_tokens(model, test_loader, device, os.path.join(args.out_dir, "visit_tokens_test.npz"))
        print("Saved learned dense visit tokens as NPZ files.")


if __name__ == "__main__":
    main()
