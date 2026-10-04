"""Experimental E37S measurement-validity matrices; never changes frozen v2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BRIGHT = ["bright_frac", "bright_count_per_Mpx", "bright_d50", "bright_d90",
          "bright_circ", "bright_d10", "bright_solidity", "bright_max_d"]
PORE = ["pore_frac", "pore_d50", "pore_elong", "pore_max_d", "crack_frac",
        "crack_count_per_Mpx", "pore_d90", "pore_count_per_Mpx"]
PARTICLE_INLENS = ["inlens_particle_texture", "inlens_particle_texture_p90",
                   "inlens_speckled_particle_frac"]
JOINT = ["etd_crack_density_particles", "etd_crack_density_graphite", "crack_g_0.05",
         "crack_g_0.1", "crack_g_0.2", "crack_g_0.3", "ridge_p97", "inlens_grad_energy"]
ANCHORS = ["corr_len_px", "fft_slope"]
FEATURES = ["pore_frac", "pore_d50", "pore_elong", "pore_max_d", "crack_frac", "crack_count_per_Mpx",
            "bright_frac", "bright_count_per_Mpx", "bright_d50", "bright_d90", "bright_circ",
            "corr_len_px", "fft_slope", "etd_crack_density_particles", "pore_d90", "pore_count_per_Mpx",
            "bright_d10", "bright_solidity", "bright_max_d", "etd_crack_density_graphite",
            "crack_g_0.05", "crack_g_0.1", "crack_g_0.2", "crack_g_0.3", "ridge_p97",
            "inlens_particle_texture", "inlens_particle_texture_p90", "inlens_speckled_particle_frac",
            "inlens_grad_energy"]
VARIANTS = ["legacy_matched", "direct_mask_gate", "dependency_gate", "quality_flags_only"]
C_GRID = (0.1, 0.5, 2.0)
VERSION = "quality-policy-audit-E37S-v1"


def bright_bad(value):
    if pd.isna(value):
        return True, "bright_quality_unknown"
    if isinstance(value, str):
        if value.lower() not in ("true", "false"):
            raise ValueError(f"Unsupported quality flag: {value!r}")
        value = value.lower() == "true"
    elif not isinstance(value, (bool, np.bool_)):
        if not isinstance(value, (int, float, np.integer, np.floating)) or value not in (0, 1):
            raise ValueError(f"Unsupported quality flag: {value!r}")
        value = bool(value)
    return bool(value), "bright_low_contrast" if value else ""


def quality(df):
    bright_values = df["bright_low_contrast"] if "bright_low_contrast" in df else pd.Series(np.nan, index=df.index)
    b = [bright_bad(value) for value in bright_values]
    raw = df["bse_p1"] if "bse_p1" in df else pd.Series(np.nan, index=df.index)
    p1 = pd.to_numeric(raw, errors="coerce").to_numpy(float)
    unknown = ~np.isfinite(p1) | (p1 < 0) | (p1 > 255)
    pore_bad = unknown | (p1 > 10)
    return pd.DataFrame(dict(bright_bad=[v[0] for v in b], bright_reason=[v[1] for v in b],
                             pore_bad=pore_bad,
                             pore_reason=np.where(unknown, "pore_quality_unknown", np.where(pore_bad, "raised_black_level", ""))),
                        index=df.index)


def prepare(df, variant="dependency_gate"):
    if variant not in VARIANTS:
        raise ValueError(variant)
    q = quality(df)
    if variant == "quality_flags_only":
        names = ["quality_bright_bad_or_unknown", "quality_pore_bad_or_unknown"]
        X = q[["bright_bad", "pore_bad"]].to_numpy(float)
        return X, names, pd.DataFrame(), q
    missing = set(FEATURES) - set(df)
    if missing:
        raise ValueError(f"Missing declared measurements: {sorted(missing)}")
    X = df[FEATURES].to_numpy(float).copy()
    X[~np.isfinite(X)] = np.nan
    audits = []
    for j, feature in enumerate(FEATURES):
        bright_dependency = feature in BRIGHT + PARTICLE_INLENS
        pore_dependency = feature in PORE
        if variant == "dependency_gate" and feature in JOINT:
            bright_dependency = pore_dependency = True
        if variant == "direct_mask_gate" and feature == "etd_crack_density_particles":
            bright_dependency = True
        for i in range(len(df)):
            reasons = []
            if variant != "legacy_matched":
                if bright_dependency and q.iloc[i].bright_bad:
                    reasons.append(q.iloc[i].bright_reason)
                if pore_dependency and q.iloc[i].pore_bad:
                    reasons.append(q.iloc[i].pore_reason)
            finite = bool(np.isfinite(X[i, j]))
            if not finite:
                reasons.append("measurement_unavailable")
            if reasons:
                X[i, j] = np.nan
            audits.append(dict(site=str(df.iloc[i].get("site", i)), feature=feature, variant=variant,
                               raw_value=float(df.iloc[i][feature]), raw_finite=finite,
                               policy_eligible=not any(r != "measurement_unavailable" for r in reasons),
                               observed_model_input=bool(np.isfinite(X[i, j])), reason="|".join(reasons)))
    return X, list(FEATURES), pd.DataFrame(audits), q


def shared_order(df):
    X = df[ANCHORS].to_numpy(float)
    if not np.isfinite(X).all() or pd.DataFrame(X).duplicated().any():
        raise ValueError("Shared mask-independent ordering requires finite unique anchor pairs")
    keys = [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest() for row in X]
    return np.argsort(keys, kind="stable")


def pipeline(C, seed=0):
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)),
                     ("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                                class_weight="balanced", max_iter=2000, random_state=seed))])


def proba(pipe, X, n_classes=3):
    out = np.zeros((len(X), n_classes))
    out[:, pipe["lr"].classes_.astype(int)] = pipe.predict_proba(X)
    return out


def balanced_logloss(y, P):
    return float(np.mean([-np.log(np.clip(P[y == c, c], 1e-12, 1)).mean() for c in np.unique(y)]))


def select_C(X, y, seed=0):
    counts = np.bincount(y)
    n_min = int(counts[counts > 0].min())
    if n_min < 2:
        return C_GRID[1]
    split = StratifiedKFold(n_splits=min(3, n_min), shuffle=True, random_state=seed)
    losses = np.zeros(len(C_GRID))
    for train, test in split.split(X, y):
        for j, C in enumerate(C_GRID):
            model = pipeline(C, seed).fit(X[train], y[train])
            losses[j] += balanced_logloss(y[test], proba(model, X[test])) * len(test)
    return C_GRID[int(losses.argmin())]


def nested_loo(X, y, seed=0, keep_models=False):
    P = np.zeros((len(y), 3)); choices = []; models = []
    for i in range(len(y)):
        train = np.flatnonzero(np.arange(len(y)) != i)
        C = select_C(X[train], y[train], seed)
        model = pipeline(C, seed).fit(X[train], y[train])
        P[i] = proba(model, X[i:i + 1])[0]
        choices.append(C)
        if keep_models:
            models.append(model)
    return P, choices, models


def balanced_accuracy(y, P):
    pred = P.argmax(axis=1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


@dataclass
class AuditModel:
    variant: str
    feature_names: list
    pipeline: Pipeline
    C: float
    classes: list
    train_observed_counts: list
    version: str = VERSION

    def predict(self, frame):
        X, names, audit, flags = prepare(frame, self.variant)
        if names != self.feature_names:
            raise ValueError("Audit feature positions changed")
        return proba(self.pipeline, X), audit, flags


def fit_model(frame, variant, classes=None):
    classes = ["Batch_1", "Batch_2", "Batch_3"] if classes is None else list(classes)
    X, names, _, _ = prepare(frame, variant)
    y = frame.batch.map({name:i for i, name in enumerate(classes)}).to_numpy(int)
    order = shared_order(frame); X, y = X[order], y[order]
    C = select_C(X, y)
    return AuditModel(variant, names, pipeline(C).fit(X, y), C, classes,
                      np.isfinite(X).sum(axis=0).tolist())
