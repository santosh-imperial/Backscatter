"""E39S bounded, fold-local models. Experimental; never loads a submission fallback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.quality_policy_audit import policy as qp

CLASSES = ["Batch_1", "Batch_2", "Batch_3"]
GRID = (0.1, 0.5, 2.0)
VERSION = "E39S-development-v1"
CANDIDATES = {
    "legacy_l1": ("legacy", "l1"),
    "gated_l1": ("gated", "l1"),
    "gated_l2": ("gated", "l2"),
    "gated_lda": ("gated", "lda"),
    "appearance_l2": ("appearance", "l2"),
    "augmented_l2": ("augmented", "l2"),
}


def matrix(frame, view, appearance_names=()):
    if view not in {"legacy", "gated", "appearance", "augmented"}:
        raise ValueError(view)
    parts, names = [], []
    if view in {"legacy", "gated", "augmented"}:
        X, old_names, _, _ = qp.prepare(frame, "legacy_matched" if view == "legacy" else "dependency_gate")
        parts.append(X); names.extend(old_names)
    if view in {"appearance", "augmented"}:
        if not appearance_names or len(set(appearance_names)) != len(appearance_names):
            raise ValueError("Require explicit, unique appearance feature names")
        forbidden = set(qp.FEATURES + ["site", "batch", "H", "W", "bright_low_contrast", "bse_p1"])
        if forbidden.intersection(appearance_names):
            raise ValueError("Appearance names overlap metadata/existing measurements")
        X = frame[list(appearance_names)].to_numpy(float).copy()
        X[~np.isfinite(X)] = np.nan
        parts.append(X); names.extend(appearance_names)
    return np.concatenate(parts, axis=1), names


def pipeline(kind, C=0.5, seed=0):
    if kind == "l1":
        classifier = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                        C=C, random_state=seed, max_iter=2000)
    elif kind == "l2":
        classifier = LogisticRegression(penalty="l2", solver="lbfgs", multi_class="multinomial",
                                        class_weight="balanced", C=C, random_state=seed, max_iter=4000)
    elif kind == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.ones(3)/3)
    else:
        raise ValueError(kind)
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)),
                     ("scale", StandardScaler()), ("classifier", classifier)])


def probabilities(model, X):
    classes = model["classifier"].classes_.astype(int)
    if set(classes) != {0, 1, 2}:
        raise ValueError("Training fold must contain all three declared classes")
    P = np.zeros((len(X), 3))
    P[:, classes] = model.predict_proba(X)
    if not np.isfinite(P).all() or not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Invalid model scores")
    return P


def select_C(X, y, kind, seed=0):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Training fold lacks a declared class")
    if kind == "lda":
        return None
    n_min = int(np.bincount(y, minlength=3).min())
    if n_min < 2:
        return 0.5
    split = StratifiedKFold(n_splits=min(3, n_min), shuffle=True, random_state=seed)
    losses = np.zeros(len(GRID))
    # Match E37S/v2's weighted fold-wise balanced-loss selection exactly.
    for train, test in split.split(X, y):
        for j, C in enumerate(GRID):
            fitted = pipeline(kind, C, seed).fit(X[train], y[train])
            losses[j] += qp.balanced_logloss(y[test], probabilities(fitted, X[test])) * len(test)
    return GRID[int(losses.argmin())]


def nested_loo(X, y, kind, seed=0):
    P = np.zeros((len(y), 3)); choices, models = [], []
    for i in range(len(y)):
        train = np.flatnonzero(np.arange(len(y)) != i)
        C = select_C(X[train], y[train], kind, seed)
        fitted = pipeline(kind, C, seed).fit(X[train], y[train])
        P[i] = probabilities(fitted, X[i:i+1])[0]
        choices.append(C); models.append(fitted)
    return P, choices, models


def metrics(y, P):
    pred = P.argmax(axis=1)
    recall = [(pred[y == c] == c).mean() for c in range(3)]
    onehot = np.eye(3)[y]
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    return dict(n_sites=len(y), n_correct=int((pred == y).sum()), accuracy=float((pred == y).mean()),
                balanced_accuracy=float(np.mean(recall)), **{f"recall_{c}":float(recall[j]) for j,c in enumerate(CLASSES)},
                balanced_logloss=qp.balanced_logloss(y, P),
                balanced_brier=float(np.mean([((P[y == c]-onehot[y == c])**2).sum(axis=1).mean() for c in range(3)])),
                confusion=confusion.tolist())


def margin_contributions(model, X, names, chosen, other):
    """Exact linear decision-function margin; uncalibrated and noncausal."""
    z = model["scale"].transform(model["impute"].transform(X))
    clf = model["classifier"]
    lookup = {int(c):j for j,c in enumerate(clf.classes_)}
    a, b = lookup[int(chosen)], lookup[int(other)]
    contributions = z[0] * (clf.coef_[a] - clf.coef_[b])
    intercept = float(clf.intercept_[a] - clf.intercept_[b])
    actual = float(clf.decision_function(X=z)[0, a] - clf.decision_function(X=z)[0, b])
    if not np.isclose(contributions.sum()+intercept, actual, atol=1e-9):
        raise AssertionError("Linear margin attribution failed")
    rows = [dict(feature=n, contribution=float(v), standardised_input=float(z[0,j]),
                 observed=bool(np.isfinite(X[0,j])), acquisition_sensitive=True,
                 family="phase-mask-dependent" if n in qp.FEATURES and n not in qp.ANCHORS else "image appearance")
            for j,(n,v) in enumerate(zip(names, contributions))]
    return sorted(rows, key=lambda row:abs(row["contribution"]), reverse=True), intercept, actual


@dataclass
class Candidate:
    name: str
    appearance_names: list
    feature_names: list
    pipeline: Pipeline
    C: float | None
    version: str = VERSION
    classes: tuple = tuple(CLASSES)

    def predict(self, frame):
        view, _ = CANDIDATES[self.name]
        X, names = matrix(frame, view, self.appearance_names)
        if names != self.feature_names:
            raise ValueError("Candidate feature order changed")
        return probabilities(self.pipeline, X)


def fit(frame, name, appearance_names=()):
    view, kind = CANDIDATES[name]
    order = qp.shared_order(frame)
    frame = frame.iloc[order].reset_index(drop=True)
    X, names = matrix(frame, view, appearance_names)
    y = frame.batch.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int)
    C = select_C(X, y, kind)
    fitted = pipeline(kind, C).fit(X, y)
    return Candidate(name, list(appearance_names), names, fitted, C)
