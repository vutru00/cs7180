"""Demographic bias metrics for evaluating generated images."""

import numpy as np


def _filter_nan(values):
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def demographic_parity_gap(scores):
    """Absolute deviation of mean gender score from 0.5.

    Args:
        scores: List of male-probability values in [0, 1]. NaN excluded.

    Returns:
        float in [0, 0.5] where 0 = perfect parity.
    """
    valid = _filter_nan(scores)
    if len(valid) == 0:
        return float("nan")
    return float(abs(np.mean(valid) - 0.5))


def representation_ratio(scores, threshold=0.5):
    """Ratio P(male) / P(female) based on a threshold.

    Args:
        scores: List of male-probability values. NaN excluded.
        threshold: Scores above this are counted as male.

    Returns:
        float (inf if no females detected).
    """
    valid = _filter_nan(scores)
    if len(valid) == 0:
        return float("nan")
    n_male = np.sum(valid > threshold)
    n_female = np.sum(valid <= threshold)
    if n_female == 0:
        return float("inf")
    return float(n_male / n_female)


def age_stats(ages):
    """Summary statistics for predicted ages.

    Args:
        ages: List of age floats. NaN excluded.

    Returns:
        Dict with keys: mean, std, frac_under_35.
    """
    valid = _filter_nan(ages)
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan"), "frac_under_35": float("nan")}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "frac_under_35": float(np.mean(valid < 35)),
    }


def detection_rate(detected):
    """Fraction of images where MiVOLO found a valid detection.

    Args:
        detected: List of booleans.
    """
    if not detected:
        return 0.0
    return float(sum(detected) / len(detected))
