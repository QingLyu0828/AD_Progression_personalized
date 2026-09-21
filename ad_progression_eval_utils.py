#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared leakage-safe evaluation utilities for AD progression experiments.

This module is intentionally dependency-light. It uses scikit-learn for binary
horizon metrics and optional scikit-survival / lifelines for survival metrics.
All scripts in the revised package use these helpers so C-index, horizon labels,
threshold selection, calibration, and Brier/IBS are computed consistently.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HAS_SKSURV = False
try:
    from sksurv.util import Surv
    from sksurv.metrics import (
        concordance_index_censored,
        cumulative_dynamic_auc,
        brier_score as sksurv_brier_score,
        integrated_brier_score,
    )
    HAS_SKSURV = True
except Exception:
    HAS_SKSURV = False

HAS_LIFELINES = False
try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False


def normalize_list_arg(x: Optional[Any]) -> List[str]:
    """Accept None, a comma-separated string, or an argparse nargs list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: List[str] = []
        for v in x:
            out.extend(normalize_list_arg(v))
        return out
    s = str(x).strip()
    if not s:
        return []
    return [v.strip() for v in s.replace(",", " ").split() if v.strip()]


def make_horizon_label(event: np.ndarray, duration: np.ndarray, horizon_years: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label whether conversion happened by a horizon.

    Included examples:
      - positives: event == 1 and duration <= horizon
      - known negatives: duration >= horizon
    Excluded examples:
      - censored before the horizon, because their status by the horizon is unknown
    """
    event = np.asarray(event).astype(int)
    duration = np.asarray(duration).astype(float)
    valid_time = np.isfinite(duration)
    positive = (event == 1) & (duration <= float(horizon_years))
    known_negative = duration >= float(horizon_years)
    include = valid_time & (positive | known_negative)
    y_horizon = positive.astype(int)
    return include, y_horizon


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, strategy: str = "youden") -> float:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    valid = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[valid]
    prob = prob[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    if strategy == "fixed_0.5":
        return 0.5
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, prob)
        score = tpr - fpr
        idx = int(np.nanargmax(score))
        thr = float(thresholds[idx])
        return thr if np.isfinite(thr) else 0.5
    if strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, prob)
        if len(thresholds) == 0:
            return 0.5
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        idx = int(np.nanargmax(f1))
        return float(thresholds[idx])
    raise ValueError(f"Unknown threshold_strategy: {strategy}")


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    valid = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[valid]
    prob = prob[valid]
    pred = (prob >= float(threshold)).astype(int)
    out: Dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "threshold": float(threshold),
    }
    if len(y_true) >= 2 and len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auc"] = out["auroc"]
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["auprc"] = out["pr_auc"]
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    else:
        out["auroc"] = np.nan
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
        out["auprc"] = np.nan
        out["balanced_accuracy"] = np.nan
    out["accuracy"] = float(accuracy_score(y_true, pred)) if len(y_true) else np.nan
    out["precision"] = float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan
    out["recall_sensitivity"] = float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan
    out["sensitivity"] = out["recall_sensitivity"]
    out["f1"] = float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan
    if len(y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        out["specificity"] = float(tn / max(tn + fp, 1))
        out["brier_score"] = float(brier_score_loss(y_true, np.clip(prob, 0.0, 1.0)))
        out["brier"] = out["brier_score"]
    else:
        out.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0, "specificity": np.nan, "brier_score": np.nan, "brier": np.nan})
    return out


def calibrate_risk(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit calibration on validation data and apply it to validation/test risks."""
    method = (method or "none").lower()
    p_val = np.asarray(p_val, dtype=float)
    p_test = np.asarray(p_test, dtype=float)
    y_val = np.asarray(y_val, dtype=int)
    info: Dict[str, Any] = {"calibration_method": method}
    valid = np.isfinite(p_val) & np.isfinite(y_val)
    if method in {"none", "raw", "identity"}:
        info["calibration_status"] = "identity_requested"
        return np.clip(p_val, 0.0, 1.0), np.clip(p_test, 0.0, 1.0), info
    if valid.sum() < 5 or len(np.unique(y_val[valid])) < 2:
        info["calibration_status"] = "identity_insufficient_validation_classes"
        return np.clip(p_val, 0.0, 1.0), np.clip(p_test, 0.0, 1.0), info
    pv = np.clip(p_val[valid], 0.0, 1.0)
    yv = y_val[valid]
    try:
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pv, yv)
            info["calibration_status"] = "fit"
            return np.clip(iso.predict(np.clip(p_val, 0.0, 1.0)), 0.0, 1.0), np.clip(iso.predict(np.clip(p_test, 0.0, 1.0)), 0.0, 1.0), info
        if method in {"platt", "logistic"}:
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(pv.reshape(-1, 1), yv)
            info["calibration_status"] = "fit"
            return lr.predict_proba(np.clip(p_val, 0.0, 1.0).reshape(-1, 1))[:, 1], lr.predict_proba(np.clip(p_test, 0.0, 1.0).reshape(-1, 1))[:, 1], info
        raise ValueError(f"Unknown calibration method: {method}")
    except Exception as e:
        info["calibration_status"] = "identity_fit_failed"
        info["calibration_error"] = str(e)
        return np.clip(p_val, 0.0, 1.0), np.clip(p_test, 0.0, 1.0), info


def fallback_c_index(events: np.ndarray, durations: np.ndarray, risk_scores: np.ndarray) -> float:
    events = np.asarray(events).astype(int)
    durations = np.asarray(durations).astype(float)
    risk_scores = np.asarray(risk_scores).astype(float)
    n = len(events)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(durations[i]) and np.isfinite(durations[j]) and np.isfinite(risk_scores[i]) and np.isfinite(risk_scores[j])):
                continue
            if durations[i] == durations[j]:
                continue
            if events[i] == 1 and durations[i] < durations[j]:
                permissible += 1
                concordant += 1 if risk_scores[i] > risk_scores[j] else 0.5 if risk_scores[i] == risk_scores[j] else 0
            elif events[j] == 1 and durations[j] < durations[i]:
                permissible += 1
                concordant += 1 if risk_scores[j] > risk_scores[i] else 0.5 if risk_scores[i] == risk_scores[j] else 0
    return float(concordant / permissible) if permissible > 0 else np.nan


def make_surv_struct(events: np.ndarray, durations: np.ndarray):
    if not HAS_SKSURV:
        raise RuntimeError("scikit-survival is not installed")
    return Surv.from_arrays(event=np.asarray(events).astype(bool), time=np.asarray(durations).astype(float))


def _as_bin_endpoints(bin_edges_or_endpoints: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(bin_edges_or_endpoints, dtype=float).ravel()
    if len(arr) == n_bins + 1:
        return arr[1:]
    if len(arr) == n_bins:
        return arr
    raise ValueError(f"Expected {n_bins} bin endpoints or {n_bins + 1} bin edges, got {len(arr)} values.")


def hazards_to_survival_by_bins(hazards: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=float)
    hazards = np.clip(hazards, 1e-7, 1.0 - 1e-7)
    return np.cumprod(1.0 - hazards, axis=1)


def risk_at_times(hazards: np.ndarray, bin_edges_or_endpoints: np.ndarray, times: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=float)
    endpoints = _as_bin_endpoints(bin_edges_or_endpoints, hazards.shape[1])
    surv = hazards_to_survival_by_bins(hazards)
    times = np.asarray(times, dtype=float).ravel()
    out = np.zeros((hazards.shape[0], len(times)), dtype=float)
    for j, t in enumerate(times):
        idx = np.searchsorted(endpoints, float(t), side="left")
        idx = int(np.clip(idx, 0, hazards.shape[1] - 1))
        out[:, j] = 1.0 - surv[:, idx]
    return np.clip(out, 0.0, 1.0)


def survival_at_times_from_hazards(hazards: np.ndarray, bin_edges_or_endpoints: np.ndarray, times: np.ndarray) -> np.ndarray:
    hazards = np.asarray(hazards, dtype=float)
    endpoints = _as_bin_endpoints(bin_edges_or_endpoints, hazards.shape[1])
    surv = hazards_to_survival_by_bins(hazards)
    times = np.asarray(times, dtype=float).ravel()
    out = np.ones((hazards.shape[0], len(times)), dtype=float)
    for j, t in enumerate(times):
        idx = np.searchsorted(endpoints, float(t), side="left")
        idx = int(np.clip(idx, 0, hazards.shape[1] - 1))
        out[:, j] = surv[:, idx]
    return np.clip(out, 0.0, 1.0)


def get_valid_eval_times(train_durations: np.ndarray, test_durations: np.ndarray, candidate_times: np.ndarray) -> np.ndarray:
    candidate_times = np.asarray(candidate_times, dtype=float)
    candidate_times = candidate_times[np.isfinite(candidate_times)]
    if len(candidate_times) == 0:
        return candidate_times
    train_durations = np.asarray(train_durations, dtype=float)
    test_durations = np.asarray(test_durations, dtype=float)
    train_durations = train_durations[np.isfinite(train_durations)]
    test_durations = test_durations[np.isfinite(test_durations)]
    if len(train_durations) == 0 or len(test_durations) == 0:
        return np.asarray([], dtype=float)
    upper = min(float(np.max(train_durations)), float(np.max(test_durations))) - 1e-6
    lower = max(0.0, min(float(np.min(train_durations)), float(np.min(test_durations)))) + 1e-6
    return np.unique(candidate_times[(candidate_times > lower) & (candidate_times < upper)])


def standard_evaluate_survival_predictions(
    model_name: str,
    train_events: np.ndarray,
    train_durations: np.ndarray,
    val_events: np.ndarray,
    val_durations: np.ndarray,
    val_hazards: np.ndarray,
    test_events: np.ndarray,
    test_durations: np.ndarray,
    test_hazards: np.ndarray,
    bin_edges_or_endpoints: np.ndarray,
    horizons: Iterable[float],
    overall_risk_horizon: Optional[float] = None,
    threshold_strategy: str = "youden",
    calibration_method: str = "isotonic",
    n_eval_grid: int = 50,
    calibration_bins: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Create a standardized set of survival/horizon metric tables."""
    horizons = [float(h) for h in horizons]
    overall_risk_horizon = float(overall_risk_horizon if overall_risk_horizon is not None else max(horizons))
    train_events = np.asarray(train_events).astype(int)
    val_events = np.asarray(val_events).astype(int)
    test_events = np.asarray(test_events).astype(int)
    train_durations = np.asarray(train_durations).astype(float)
    val_durations = np.asarray(val_durations).astype(float)
    test_durations = np.asarray(test_durations).astype(float)

    val_risk_overall = risk_at_times(val_hazards, bin_edges_or_endpoints, np.asarray([overall_risk_horizon]))[:, 0]
    test_risk_overall = risk_at_times(test_hazards, bin_edges_or_endpoints, np.asarray([overall_risk_horizon]))[:, 0]

    summary: Dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "n_test": int(len(test_events)),
        "n_test_events": int(np.nansum(test_events)),
        "overall_risk_horizon_years": overall_risk_horizon,
        "calibration_method": calibration_method,
        "has_sksurv": HAS_SKSURV,
    }
    if HAS_SKSURV:
        try:
            ci = concordance_index_censored(test_events.astype(bool), test_durations.astype(float), test_risk_overall.astype(float))[0]
            summary["c_index"] = float(ci)
        except Exception as e:
            summary["c_index"] = np.nan
            summary["c_index_error"] = str(e)
    else:
        summary["c_index"] = fallback_c_index(test_events, test_durations, test_risk_overall)

    # Overall binary metric at the overall horizon using calibrated risk.
    inc_val_o, y_val_o = make_horizon_label(val_events, val_durations, overall_risk_horizon)
    inc_test_o, y_test_o = make_horizon_label(test_events, test_durations, overall_risk_horizon)
    if inc_test_o.sum() >= 5 and len(np.unique(y_test_o[inc_test_o])) >= 2:
        p_val_cal, p_test_cal, cal_info = calibrate_risk(y_val_o[inc_val_o], val_risk_overall[inc_val_o], test_risk_overall[inc_test_o], calibration_method)
        thr = choose_threshold(y_val_o[inc_val_o], p_val_cal, threshold_strategy) if inc_val_o.sum() >= 5 and len(np.unique(y_val_o[inc_val_o])) >= 2 else 0.5
        overall_bin = compute_binary_metrics(y_test_o[inc_test_o], p_test_cal, thr)
        for k, v in overall_bin.items():
            summary[f"overall_horizon_{k}"] = v
        summary.update({f"overall_horizon_{k}": v for k, v in cal_info.items()})

    # Time-dependent AUC and Brier/IBS on raw survival curves. Calibration of an
    # entire survival curve requires a separate calibration model and is not
    # applied here; horizon-level Brier is calibrated above/below.
    tda_rows: List[Dict[str, Any]] = []
    brier_rows: List[Dict[str, Any]] = []
    if HAS_SKSURV:
        try:
            y_train_surv = make_surv_struct(train_events, train_durations)
            y_test_surv = make_surv_struct(test_events, test_durations)
            valid_horizon_times = get_valid_eval_times(train_durations, test_durations, np.asarray(horizons, dtype=float))
            if len(valid_horizon_times) > 0:
                risk_mat = risk_at_times(test_hazards, bin_edges_or_endpoints, valid_horizon_times)
                aucs, mean_auc = cumulative_dynamic_auc(y_train_surv, y_test_surv, risk_mat, valid_horizon_times)
                for t, auc in zip(valid_horizon_times, aucs):
                    tda_rows.append({"model": model_name, "seed": seed, "time_years": float(t), "time_dependent_auc": float(auc)})
                summary["mean_time_dependent_auc"] = float(mean_auc)
        except Exception as e:
            summary["time_dependent_auc_error"] = str(e)
        try:
            n_bins = np.asarray(test_hazards).shape[1]
            endpoints = _as_bin_endpoints(np.asarray(bin_edges_or_endpoints), n_bins)
            max_eval = min(float(np.nanmax(train_durations)), float(np.nanmax(test_durations)), float(np.nanmax(endpoints))) - 1e-6
            min_eval = max(1e-3, float(np.nanpercentile(test_durations[np.isfinite(test_durations)], 5)))
            if max_eval > min_eval:
                eval_times = np.linspace(min_eval, max_eval, int(n_eval_grid))
                eval_times = get_valid_eval_times(train_durations, test_durations, eval_times)
                if len(eval_times) >= 2:
                    surv_mat = survival_at_times_from_hazards(test_hazards, bin_edges_or_endpoints, eval_times)
                    times_bs, bs = sksurv_brier_score(y_train_surv, y_test_surv, surv_mat, eval_times)
                    for t, b in zip(times_bs, bs):
                        brier_rows.append({"model": model_name, "seed": seed, "time_years": float(t), "brier_score": float(b)})
                    summary["integrated_brier_score"] = float(integrated_brier_score(y_train_surv, y_test_surv, surv_mat, eval_times))
        except Exception as e:
            summary["brier_ibs_error"] = str(e)

    horizon_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    for h in horizons:
        val_risk = risk_at_times(val_hazards, bin_edges_or_endpoints, np.asarray([h]))[:, 0]
        test_risk = risk_at_times(test_hazards, bin_edges_or_endpoints, np.asarray([h]))[:, 0]
        val_include, y_val = make_horizon_label(val_events, val_durations, h)
        test_include, y_test = make_horizon_label(test_events, test_durations, h)
        base_row = {
            "model": model_name,
            "seed": seed,
            "horizon_years": float(h),
            "n_included": int(test_include.sum()),
            "n_positive": int(y_test[test_include].sum()) if test_include.sum() else 0,
            "calibration_method": calibration_method,
        }
        if test_include.sum() < 5 or len(np.unique(y_test[test_include])) < 2:
            base_row.update({"skipped": True, "reason": "too_few_cases_or_single_class"})
            horizon_rows.append(base_row)
            continue
        if val_include.sum() >= 5 and len(np.unique(y_val[val_include])) >= 2:
            p_val, p_test, cal_info = calibrate_risk(y_val[val_include], val_risk[val_include], test_risk[test_include], calibration_method)
            threshold = choose_threshold(y_val[val_include], p_val, threshold_strategy)
        else:
            p_test = np.clip(test_risk[test_include], 0.0, 1.0)
            threshold = 0.5
            cal_info = {"calibration_method": calibration_method, "calibration_status": "identity_insufficient_validation_classes"}
        hm = compute_binary_metrics(y_test[test_include], p_test, threshold)
        hm.update(base_row)
        hm.update(cal_info)
        hm["raw_brier_score"] = float(brier_score_loss(y_test[test_include], np.clip(test_risk[test_include], 0.0, 1.0)))
        hm["skipped"] = False
        horizon_rows.append(hm)

        # Calibration bins after calibration.
        try:
            pred = np.asarray(p_test, dtype=float)
            yobs = y_test[test_include]
            qbins = pd.qcut(pred, q=int(calibration_bins), duplicates="drop")
            calib_df = pd.DataFrame({"pred": pred, "obs": yobs, "bin": qbins})
            grouped = calib_df.groupby("bin", observed=True)
            for bidx, (_, gg) in enumerate(grouped):
                calib_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "horizon_years": float(h),
                    "bin": int(bidx),
                    "n": int(len(gg)),
                    "mean_predicted_risk": float(gg["pred"].mean()),
                    "observed_event_rate": float(gg["obs"].mean()),
                    "absolute_calibration_error": float(abs(gg["pred"].mean() - gg["obs"].mean())),
                    "calibration_method": calibration_method,
                })
        except Exception:
            pass

    km_rows: List[Dict[str, Any]] = []
    if HAS_LIFELINES:
        try:
            risk = np.asarray(test_risk_overall, dtype=float)
            med = np.nanmedian(risk)
            high = risk >= med
            low = ~high
            if high.sum() >= 3 and low.sum() >= 3:
                lr = logrank_test(
                    test_durations[high], test_durations[low],
                    event_observed_A=test_events[high], event_observed_B=test_events[low],
                )
                hr = np.nan
                try:
                    df_hr = pd.DataFrame({"duration": test_durations, "event": test_events, "high_risk": high.astype(int)})
                    cph = CoxPHFitter()
                    cph.fit(df_hr, duration_col="duration", event_col="event")
                    hr = float(np.exp(cph.params_["high_risk"]))
                except Exception:
                    pass
                km_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "risk_horizon_years": overall_risk_horizon,
                    "risk_threshold_median": float(med),
                    "n_high_risk": int(high.sum()),
                    "n_low_risk": int(low.sum()),
                    "events_high_risk": int(test_events[high].sum()),
                    "events_low_risk": int(test_events[low].sum()),
                    "logrank_p_value": float(lr.p_value),
                    "hazard_ratio_high_vs_low": hr,
                })
        except Exception as e:
            km_rows.append({"model": model_name, "seed": seed, "error": str(e)})
    else:
        km_rows.append({"model": model_name, "seed": seed, "error": "lifelines_not_installed"})

    return {
        "summary": pd.DataFrame([summary]),
        "horizon": pd.DataFrame(horizon_rows),
        "time_dependent_auc": pd.DataFrame(tda_rows),
        "brier_scores": pd.DataFrame(brier_rows),
        "calibration": pd.DataFrame(calib_rows),
        "km": pd.DataFrame(km_rows),
    }


def hazards_from_prediction_dataframe(pred: pd.DataFrame, prefix: str = "hazard_bin") -> np.ndarray:
    cols = [c for c in pred.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns starting with '{prefix}' were found.")
    def _idx(c: str) -> int:
        tail = c.replace(prefix, "")
        return int(tail) if tail.isdigit() else 10**9
    cols = sorted(cols, key=_idx)
    return pred[cols].astype(float).to_numpy()
