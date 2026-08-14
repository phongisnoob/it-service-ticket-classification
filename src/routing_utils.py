import numpy as np
import pandas as pd


def compute_bootstrap_ci(
    results_df: pd.DataFrame,
    threshold: float,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Bootstrap 95% CI for routing accuracy and coverage at a given threshold.

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

    alpha = (1.0 - ci) / 2.0
    acc_ci = np.percentile(boot_accs, [alpha * 100, (1 - alpha) * 100])
    cov_ci = np.percentile(boot_covs, [alpha * 100, (1 - alpha) * 100])
    return acc_ci.tolist(), cov_ci.tolist()
