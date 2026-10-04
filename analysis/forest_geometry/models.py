"""E41S fixed factorial; experimental crop classification, never submission fallback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from analysis.classification_m1_m2 import models as previous
from analysis.forest_geometry import geometry as g
from analysis.quality_policy_audit import policy as qp

CLASSES = previous.CLASSES
VERSION = "E41S-fixed-forest-geometry-v1"
CANDIDATES = {
    "legacy_l1": ("legacy", "l1"),
    "base_l1": ("base", "l1"),
    "base_rf": ("base", "rf"),
    "geometry_l1": ("geometry", "l1"),
    "geometry_rf": ("geometry", "rf"),
    "quality_rf": ("quality", "rf"),
    "availability_rf": ("availability", "rf"),
}
GROUPS = {
    "existing_bright": qp.BRIGHT,
    "existing_pore": qp.PORE,
    "joint_appearance": qp.JOINT,
    "particle_texture": qp.PARTICLE_INLENS,
    "raw_BSE_anchors": qp.ANCHORS,
    "new_shape": g.FEATURES[:3],
    "new_width": g.FEATURES[3:5],
    "new_spacing": [g.FEATURES[5]],
    "new_dispersion": [g.FEATURES[6]],
    "new_crossphase": [g.FEATURES[7]],
}


def group(feature):
    for name, features in GROUPS.items():
        if feature in features:
            return name
    return "acquisition_or_availability_control"


def matrix(frame, view):
    if view == "quality":
        X, names, _, _ = qp.prepare(frame, "quality_flags_only")
        return X, names
    if view not in {"legacy", "base", "geometry", "availability"}:
        raise ValueError(view)
    X, names, _, _ = qp.prepare(frame, "legacy_matched" if view == "legacy" else "dependency_gate")
    if view in {"geometry", "availability"}:
        extra, extra_names, _, _ = g.prepare(frame)
        X = np.concatenate((X, extra), axis=1)
        names += extra_names
    if view == "availability":
        X = np.isfinite(X).astype(float)
        names = ["available__" + name for name in names]
    return X, names


def pipeline(kind, C=None, seed=0):
    if kind == "l1":
        return previous.pipeline("l1", C=0.5 if C is None else C, seed=seed)
    if kind != "rf":
        raise ValueError(kind)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)),
        ("classifier", RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=3,
             min_samples_leaf=4, max_features="sqrt", class_weight="balanced", bootstrap=True,
             random_state=seed, oob_score=False, n_jobs=1)),
    ])


probabilities = previous.probabilities
metrics = previous.metrics


def select_C(X, y, kind, seed=0):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Training fold lacks a declared class")
    return previous.select_C(X, y, "l1", seed) if kind == "l1" else None


def nested_loo(X, y, kind, seed=0):
    P = np.zeros((len(y), 3))
    choices, models = [], []
    for i in range(len(y)):
        train = np.flatnonzero(np.arange(len(y)) != i)
        C = select_C(X[train], y[train], kind, seed)
        fitted = pipeline(kind, C, seed).fit(X[train], y[train])
        P[i] = probabilities(fitted, X[i:i+1])[0]
        choices.append(C)
        models.append(fitted)
    return P, choices, models


def forest_contributions(model, X, names, chosen, other):
    """Exact average-tree path score difference. Order dependent, noncausal; not SHAP."""
    if len(X) != 1 or len(names) != X.shape[1]:
        raise ValueError("Explain exactly one input with its declared feature positions")
    # sklearn tree traversal coerces to float32; use the same rounding near splits.
    z = np.asarray(model["impute"].transform(X), dtype=np.float32)[0]
    clf = model["classifier"]
    if set(clf.classes_) != {0, 1, 2}:
        raise ValueError("Require all three declared classes")
    lookup = {int(c): j for j, c in enumerate(clf.classes_)}
    a, b = lookup[int(chosen)], lookup[int(other)]
    root = np.zeros(3)
    contributions = np.zeros((len(names), 3))
    for estimator in clf.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        probabilities_at_nodes = values / values.sum(axis=1, keepdims=True)
        root += probabilities_at_nodes[0]
        node = 0
        while tree.children_left[node] != tree.children_right[node]:
            feature = int(tree.feature[node])
            child = tree.children_left[node] if z[feature] <= tree.threshold[node] else tree.children_right[node]
            contributions[feature] += probabilities_at_nodes[child] - probabilities_at_nodes[node]
            node = child
    root /= len(clf.estimators_)
    contributions /= len(clf.estimators_)
    score = model.predict_proba(X)[0]
    if not np.allclose(root + contributions.sum(axis=0), score, atol=1e-12, rtol=0):
        raise AssertionError("Forest path decomposition does not reconstruct predict_proba")
    root_margin = float(root[a] - root[b])
    actual = float(score[a] - score[b])
    rows = [dict(feature=name, contribution=float(contributions[j, a] - contributions[j, b]),
                 observed=bool(np.isfinite(X[0, j])), imputed_input=float(z[j]))
            for j, name in enumerate(names)]
    return sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True), root_margin, actual


def explain(model, X, names, chosen, other, kind):
    if kind == "rf":
        rows, root, margin = forest_contributions(model, X, names, chosen, other)
        explanation_type = "forest_path_score_margin"
    else:
        rows, root, margin = previous.margin_contributions(model, X, names, chosen, other)
        explanation_type = "linear_decision_margin"
    for row in rows:
        row.update(feature_group=group(row["feature"]), input_value=float(X[0, names.index(row["feature"])]),
                   explanation_type=explanation_type)
    return rows, root, margin


def preserve_nominal_availability(nominal, changed):
    if nominal.shape != changed.shape:
        raise ValueError("Variant must preserve all input positions")
    changed = changed.copy()
    changed[~np.isfinite(nominal)] = np.nan
    return changed


def replace_observed_group(model, X, names, features):
    """Fixed-model train-median replacement; preserve unavailable cells."""
    out = X.copy()
    indices = [names.index(name) for name in features]
    medians = model["impute"].statistics_
    n = 0
    for j in indices:
        if np.isfinite(out[0, j]):
            out[0, j] = medians[j]
            n += 1
    return out, n


@dataclass
class Candidate:
    name: str
    feature_names: list
    pipeline: Pipeline
    C: float | None
    version: str = VERSION

    def predict(self, frame):
        X, names = matrix(frame, CANDIDATES[self.name][0])
        if names != self.feature_names:
            raise ValueError("Feature positions changed")
        return probabilities(self.pipeline, X)


def fit(frame, name):
    frame = frame.iloc[qp.shared_order(frame)].reset_index(drop=True)
    view, kind = CANDIDATES[name]
    X, names = matrix(frame, view)
    y = frame.batch.map({c:i for i, c in enumerate(CLASSES)}).to_numpy(int)
    C = select_C(X, y, kind)
    return Candidate(name, names, pipeline(kind, C).fit(X, y), C)
