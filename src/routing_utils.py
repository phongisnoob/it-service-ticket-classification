"""Routing threshold risk-control utilities.

Threshold selection uses simultaneous one-sided exact Clopper-Pearson
confidence lower bounds with a Bonferroni correction for the number of
candidate thresholds evaluated.  This replaces the previous bootstrap CI
which had no multiple-testing correction and was non-deterministic.

Reference:
  Clopper, C.J. and Pearson, E.S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. Biometrika, 26(4), 404-413.
"""

import math
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact Clopper-Pearson lower bound at significance level alpha.

    Returns 0.0 when trials == 0.
    """
    if trials == 0:
        return 0.0
    if successes == 0:
        return 0.0
    # Lower bound: alpha-th quantile of Beta(successes, trials - successes + 1)
    return float(beta_dist.ppf(alpha, successes, trials - successes + 1))


def simultaneous_lower_bound(
    successes: int,
    trials: int,
    alpha: float,
    n_candidates: int,
) -> float:
    """Simultaneous Clopper-Pearson lower bound with Bonferroni correction.

    Adjusts alpha for n_candidates to control the family-wise error rate.
    Returns 0.0 when trials == 0 or n_candidates == 0.
    """
    if n_candidates <= 0 or trials == 0:
        return 0.0
    adjusted_alpha = alpha / n_candidates
    return clopper_pearson_lower(successes, trials, adjusted_alpha)


def compute_threshold_candidates(
    results_df: pd.DataFrame,
    threshold_start: float = 0.10,
    threshold_stop: float = 1.00,
    threshold_step: float = 0.01,
    target_accuracy: float = 0.90,
    alpha: float = 0.05,
    min_accepted_samples: int = 50,
) -> pd.DataFrame:
    """Evaluate every threshold in a fixed grid and compute simultaneous bounds.

    Args:
        results_df: DataFrame with columns ``confidence`` and ``correct``.
        threshold_start: Inclusive lower bound of the threshold grid.
        threshold_stop: Exclusive upper bound of the threshold grid.
        threshold_step: Step size of the threshold grid.
        target_accuracy: Minimum required accuracy for auto-routed tickets.
        alpha: Overall family-wise significance level before Bonferroni.
        min_accepted_samples: Minimum number of auto-routed tickets required
            for a threshold to be considered valid.

    Returns:
        DataFrame with one row per threshold containing coverage, point
        accuracy, simultaneous lower bound, and a ``meets_requirement``
        boolean.
    """
    thresholds = np.round(np.arange(threshold_start, threshold_stop, threshold_step), 2)
    n_candidates = len(thresholds)
    rows = []
    n_total = len(results_df)

    for threshold in thresholds:
        routed = results_df[results_df["confidence"] >= threshold]
        n_routed = len(routed)
        coverage = n_routed / n_total if n_total > 0 else 0.0
        point_acc = float(routed["correct"].mean()) if n_routed > 0 else 0.0
        n_correct = int(routed["correct"].sum()) if n_routed > 0 else 0

        lb = (
            simultaneous_lower_bound(n_correct, n_routed, alpha, n_candidates)
            if n_routed >= min_accepted_samples
            else 0.0
        )

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "n_accepted": n_routed,
                "n_correct": n_correct,
                "coverage": coverage,
                "auto_routed_accuracy": point_acc,
                "accuracy_ci_lower": lb,
                "manual_review_rate": 1.0 - coverage,
                "meets_requirement": lb >= target_accuracy,
            }
        )

    return pd.DataFrame(rows)


def select_threshold(
    results_df: pd.DataFrame,
    threshold_start: float = 0.10,
    threshold_stop: float = 1.00,
    threshold_step: float = 0.01,
    target_accuracy: float = 0.90,
    alpha: float = 0.05,
    min_accepted_samples: int = 50,
) -> tuple[pd.Series | None, pd.DataFrame]:
    """Select the threshold that maximises coverage subject to the accuracy requirement.

    Returns ``(selected_row_or_None, full_analysis_df)``.
    """
    df = compute_threshold_candidates(
        results_df,
        threshold_start=threshold_start,
        threshold_stop=threshold_stop,
        threshold_step=threshold_step,
        target_accuracy=target_accuracy,
        alpha=alpha,
        min_accepted_samples=min_accepted_samples,
    )

    eligible = df[df["meets_requirement"]].sort_values("coverage", ascending=False)
    if eligible.empty:
        return None, df

    selected: pd.Series = eligible.iloc[0]
    return selected, df


# ---------------------------------------------------------------------------
# Legacy bootstrap API kept for any callers that have not yet been migrated.
# New code should use select_threshold / compute_threshold_candidates instead.
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    results_df: pd.DataFrame,
    threshold: float,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Bootstrap CI for routing accuracy and coverage.

    .. deprecated::
        This function is **not used** in the canonical DVC pipeline.
        The production threshold selection uses ``select_threshold`` which
        applies simultaneous exact Clopper-Pearson bounds with Bonferroni
        correction — a statistically sounder and fully deterministic approach.

        ``compute_bootstrap_ci`` is retained only for research/exploratory
        use.  Do NOT import it in pipeline scripts.

    Returns (accuracy_ci, coverage_ci) each as [lower, upper].
    """
    rng = np.random.default_rng(seed)
    boot_accs: list[float] = []
    boot_covs: list[float] = []
    n = len(results_df)

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample = results_df.iloc[indices]
        routed = sample[sample["confidence"] >= threshold]
        boot_covs.append(len(routed) / n)
        boot_accs.append(float(routed["correct"].mean()) if len(routed) > 0 else 0.0)

    half_alpha = (1.0 - ci) / 2.0
    acc_ci = np.percentile(boot_accs, [half_alpha * 100, (1 - half_alpha) * 100])
    cov_ci = np.percentile(boot_covs, [half_alpha * 100, (1 - half_alpha) * 100])
    return acc_ci.tolist(), cov_ci.tolist()


# ---------------------------------------------------------------------------
# Per-class reporting helper
# ---------------------------------------------------------------------------


class ClassAcceptedStats(TypedDict):
    label: str
    n_accepted: int
    n_correct: int
    accuracy: float
    wilson_lb_95: float


def per_class_accepted_stats(
    results_df: pd.DataFrame,
    threshold: float,
) -> list[ClassAcceptedStats]:
    """Return per-class accepted count, accuracy, and one-sided Wilson lower bound."""
    routed = results_df[results_df["confidence"] >= threshold].copy()
    stats: list[ClassAcceptedStats] = []
    for label, group in routed.groupby("true_label"):
        n = len(group)
        n_correct = int(group["correct"].sum())
        p_hat = n_correct / n if n > 0 else 0.0
        # Wilson lower bound (one-sided 95%)
        z = 1.645
        if n > 0:
            denom = 1 + z * z / n
            centre = (p_hat + z * z / (2 * n)) / denom
            margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
            wilson_lb = max(0.0, centre - margin)
        else:
            wilson_lb = 0.0
        stats.append(
            ClassAcceptedStats(
                label=str(label),
                n_accepted=n,
                n_correct=n_correct,
                accuracy=p_hat,
                wilson_lb_95=wilson_lb,
            )
        )
    return stats
