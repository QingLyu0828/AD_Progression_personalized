#!/usr/bin/env python3
"""Load a trained PROMISE-AD model and predict raw risks without outcome labels."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import train


class PreprocessorUnpickler(pickle.Unpickler):
    """Support preprocessors saved by direct execution of the original trainer."""

    def find_class(self, module, name):
        # Compatibility aliases for serialized research artifacts, not model variants.
        if module in {"__main__", "train", "promise_ad_v2_revision",
                      "progression_aware_survival_model_presplit_v2"} and name in {
            "DataConfig", "LongitudinalVisitPreprocessor"
        }:
            return getattr(train, name)
        return super().find_class(module, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Input and output paths must differ")

    config = json.loads((args.model_dir / "config.json").read_text())
    with (args.model_dir / "visit_preprocessor.pkl").open("rb") as handle:
        pre = PreprocessorUnpickler(handle).load()
    pre.config = copy.deepcopy(pre.config)
    cfg = pre.config
    frame = pd.read_csv(args.input_csv, low_memory=False)
    required = [cfg.id_col, cfg.date_col, *pre.numeric_cols, *pre.categorical_cols]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing trained input columns: {missing}")
    if frame.empty or frame[cfg.id_col].isna().any():
        raise ValueError("Input must contain visits with nonmissing subject IDs")
    frame[cfg.date_col] = pd.to_datetime(frame[cfg.date_col], errors="raise")
    if frame[cfg.date_col].isna().any():
        raise ValueError("Visit dates must not be missing")

    # The caller supplies only the intended observation window. Outcome/index dates
    # and labels are never used to select or score new inference subjects.
    cfg.strict_preindex_filter = False
    frame[cfg.duration_col] = 1.0
    frame[cfg.event_col] = 0
    ids = frame[cfg.id_col].drop_duplicates().tolist()
    # Fail on unusable subjects instead of the original dataset's warn-and-skip path.
    for rid, group in frame.groupby(cfg.id_col, sort=False):
        if pre.transform_group(group)["n_visits"] < cfg.min_visits:
            raise ValueError(f"Subject {rid!r} has fewer than {cfg.min_visits} visits")
    dataset = train.ADProgressionDataset(frame, ids, pre)
    if len(dataset) != len(ids):
        raise ValueError("One or more subjects failed preprocessing; no predictions written")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=train.collate_visit_sequences, num_workers=0)
    device = torch.device(args.device)
    bins = torch.tensor(config["train"]["bins"], dtype=torch.float32)
    horizons = torch.tensor(config["train"]["horizons"], dtype=torch.float32)
    model = train.ProgressionAwareMultiHorizonSurvivalModel(
        num_dim=3 * len(pre.numeric_cols),
        cat_vocab_sizes=[len(pre.vocabs[col]) for col in pre.categorical_cols],
        n_bins=len(bins), cfg=train.ModelConfig(**config["model"]),
    ).to(device)
    checkpoint = torch.load(args.model_dir / "best_model.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    result = train.predict(model, loader, device, bins, horizons)
    result = result.drop(columns=["duration", "event"])
    # Core predict() names the subject column RID even when a custom input ID is used.
    if cfg.id_col != "RID":
        result = result.rename(columns={"RID": cfg.id_col})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Saved raw, uncalibrated risks for {len(result)} subjects to {args.output_csv}")


if __name__ == "__main__":
    main()
