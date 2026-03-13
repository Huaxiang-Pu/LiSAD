"""Evaluation metrics for anomaly detection and attack-response timing."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def nwdd(y_true, scores, threshold):
    """Compute normalized weighted detection delay for one score sequence."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    attack_indices = np.where(y_true > 0)[0]
    if len(attack_indices) == 0:
        raise ValueError("No attack found in y_true (all zeros).")

    t0 = attack_indices[0]
    detect_indices = np.where((scores >= threshold) & (np.arange(len(scores)) >= t0))[0]
    if len(detect_indices) == 0:
        return 1.0

    td = detect_indices[0]
    total_attack = np.sum(y_true[t0:])
    detected_attack = np.sum(y_true[t0 : td + 1])

    if total_attack == 0:
        return 0.0

    return float(detected_attack / total_attack)


def ttfd(y_true, scores, threshold):
    """Compute time-to-first-detection in timesteps."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    attack_indices = np.where(y_true > 0)[0]
    if len(attack_indices) == 0:
        raise ValueError("No attack found in y_true.")

    t0 = attack_indices[0]
    detect_indices = np.where((scores >= threshold) & (np.arange(len(scores)) >= t0))[0]
    if len(detect_indices) == 0:
        return np.inf

    td = detect_indices[0]
    return int(td - t0)


def tas(
    y_true,
    scores,
    t0: int | None = None,
    t1: int | None = None,
    smooth: str | None = "ema",
    ema_alpha: float = 0.2,
    window: int = 10,
    diff: str = "first",
    method: str = "pearson",
    eps: float = 1e-12,
) -> float:
    """Measure alignment between attack-intensity and anomaly-score trends."""
    a = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if a.shape[0] != s.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true({a.shape[0]}) vs scores({s.shape[0]})"
        )

    T = a.shape[0]

    if t0 is None:
        idx = np.where(a > 0)[0]
        if idx.size == 0:
            raise ValueError(
                "No attack found in y_true (all zeros). Provide t0 explicitly if needed."
            )
        t0 = int(idx[0])

    if t1 is None:
        t1 = T
    t0 = max(0, min(T, int(t0)))
    t1 = max(0, min(T, int(t1)))
    if t1 - t0 < 3:
        return 0.0

    a_seg = a[t0:t1].copy()
    s_seg = s[t0:t1].copy()

    def ema(x: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(alpha)
        if not (0 < alpha <= 1):
            raise ValueError("ema_alpha must be in (0,1].")
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    def moving_average(x: np.ndarray, w: int) -> np.ndarray:
        w = int(w)
        if w <= 1:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    if smooth == "ema":
        a_seg = ema(a_seg, ema_alpha)
        s_seg = ema(s_seg, ema_alpha)
    elif smooth == "ma":
        a_seg = moving_average(a_seg, window)
        s_seg = moving_average(s_seg, window)
    elif smooth is not None:
        raise ValueError("smooth must be one of {'ema','ma',None}.")

    if diff == "first":
        da = np.diff(a_seg)
        ds = np.diff(s_seg)
    elif diff == "second":
        da = np.diff(a_seg, n=2)
        ds = np.diff(s_seg, n=2)
    else:
        raise ValueError("diff must be 'first' or 'second'.")

    if da.size < 2:
        return 0.0

    if method == "pearson":
        da_c = da - da.mean()
        ds_c = ds - ds.mean()
        denom = (np.sqrt((da_c**2).mean()) * np.sqrt((ds_c**2).mean())) + eps
        score = float((da_c * ds_c).mean() / denom)
        return float(np.clip(score, -1.0, 1.0))

    if method == "spearman":
        def rankdata(x: np.ndarray) -> np.ndarray:
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x), dtype=float)
            xs = x[order]
            i = 0
            while i < len(xs):
                j = i
                while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                    j += 1
                if j > i:
                    avg = 0.5 * (i + j)
                    ranks[order[i : j + 1]] = avg
                i = j + 1
            return ranks

        r_da = rankdata(da)
        r_ds = rankdata(ds)
        r_da -= r_da.mean()
        r_ds -= r_ds.mean()
        denom = (np.sqrt((r_da**2).mean()) * np.sqrt((r_ds**2).mean())) + eps
        score = float((r_da * r_ds).mean() / denom)
        return float(np.clip(score, -1.0, 1.0))

    raise ValueError("method must be 'pearson' or 'spearman'.")


def evaluate_scores(y_true, scores):
    """Compute thresholded and threshold-free anomaly-detection metrics."""
    results = {}

    y_true_bin = (y_true > 0).astype(int)
    attack_rate = y_true_bin.mean()
    threshold = np.quantile(scores, 1.0 - attack_rate)
    results["_threshold"] = threshold

    results["AUC-ROC"] = roc_auc_score(y_true_bin, scores)
    results["AUC-PR"] = average_precision_score(y_true_bin, scores)
    results["F1"] = f1_score(y_true_bin, (scores >= threshold).astype(int))

    fpr, tpr, thr = roc_curve(y_true_bin, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    results["FPR@95TPR"] = fpr[idx]

    results["TTFD"] = ttfd(y_true, scores, threshold)
    results["NWDD"] = nwdd(y_true, scores, threshold)
    results["TAS"] = tas(
        y_true,
        scores,
        t0=None,
        t1=None,
        smooth="ema",
        ema_alpha=0.2,
        diff="first",
        method="pearson",
    )

    return results
