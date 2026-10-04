"""E43S fixed scalar layer on normalized OvR scores; no class reassignment."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

BOUNDS = (0.05, 20.0)
LOSS_NAMES = ("multiclass_logloss", "balanced_multiclass_logloss", "multiclass_brier",
              "balanced_multiclass_brier", "bet_brier", "balanced_bet_brier", "bet_nll", "balanced_bet_nll")


def transform(P, alpha):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Require nonnegative normalized three-class scores")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Require positive finite inverse temperature")
    logs = np.full_like(P,-np.inf)
    np.log(P,where=P>0,out=logs)
    Q = P.copy() if alpha == 1 else softmax(alpha * logs, axis=1)
    if not np.array_equal(P.argmax(axis=1), Q.argmax(axis=1)):
        raise AssertionError("Scalar changed argmax")
    return Q


def balanced_nll(y, P):
    y = np.asarray(y, dtype=int)
    if len(y) != len(P) or set(y) != {0, 1, 2}:
        raise ValueError("Calibration requires all three training classes")
    loss = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
    return float(np.mean([loss[y == c].mean() for c in range(3)]))


def fit_scalar(y, P):
    # Validate identity first; optimize only one declared scalar, never drop labels.
    identity = balanced_nll(y, transform(P, 1.0))
    objective = lambda a: balanced_nll(y, transform(P, a))
    result = minimize_scalar(objective, bounds=BOUNDS, method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError("Scalar optimizer failed")
    choices = [(identity, 1.0)] + [(objective(a), float(a)) for a in [BOUNDS[0], result.x, BOUNDS[1]]]
    best = min(v for v, a in choices)
    alpha = next(a for v, a in choices if v <= best + 1e-10)
    return dict(alpha=alpha, temperature=1/alpha, training_balanced_nll=objective(alpha),
                identity_training_balanced_nll=identity, boundary=alpha in BOUNDS)


def promotion(original, fitted):
    primary = ["balanced_multiclass_logloss", "balanced_bet_brier"]
    reductions = {name: (original[name]-fitted[name])/original[name] for name in primary}
    nonworsening = {name: fitted[name] <= original[name]+1e-6 for name in LOSS_NAMES if name not in primary}
    passed = all(v >= .01 for v in reductions.values()) and all(nonworsening.values())
    return dict(loss_gate_passed=bool(passed), relative_primary_reductions=reductions,
                other_loss_nonworsening=nonworsening, minimum_relative_reduction=.01,
                maximum_other_loss_increase=1e-6)


def metrics(P, y, chosen):
    """Common-label losses and unchanged-bet correctness, under both weightings."""
    P, y, chosen = np.asarray(P), np.asarray(y), np.asarray(chosen)
    if P.shape != (len(y),3) or not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1),1):
        raise ValueError("Invalid three-class vectors")
    q, z = P[np.arange(len(y)),chosen], (chosen == y).astype(float)
    true = P[np.arange(len(y)),y]
    losses = dict(multiclass_logloss=-np.log(np.clip(true,1e-12,1)),
        multiclass_brier=((P-np.eye(3)[y])**2).sum(axis=1), bet_brier=(q-z)**2,
        bet_nll=-(z*np.log(np.clip(q,1e-12,1))+(1-z)*np.log(np.clip(1-q,1e-12,1))))
    weighted = lambda v: float(np.mean([v[y==c].mean() for c in np.unique(y)]))
    result = dict(n_sites=len(y),n_correct=int(z.sum()),n_wrong=int((1-z).sum()),
        accuracy=float(z.mean()),balanced_accuracy=weighted(z),mean_bet_score=float(q.mean()),
        mean_wrong_bet_score=float(q[z==0].mean()) if (z==0).any() else np.nan,
        mean_correct_bet_score=float(q[z==1].mean()) if (z==1).any() else np.nan,
        mean_true_class_score=float(true.mean()))
    for name,values in losses.items():
        result[name] = float(values.mean()); result['balanced_'+name] = weighted(values)
    return result
