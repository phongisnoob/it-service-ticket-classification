import numpy as np


def compute_bootstrap_ci(results_df, threshold, n_bootstraps=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence intervals for routing accuracy and coverage.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain 'confidence' and 'correct' columns.
    threshold : float
        Confidence threshold for auto-routing.
    n_bootstraps : int
        Number of bootstrap samples.
    ci : float
        Confidence interval level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple[list[float], list[float]]
        (accuracy_ci, coverage_ci) each as [lower, upper].
    """
    rng = np.random.default_rng(seed)
    boot_accs = []
    boot_covs = []
    n = len(results_df)

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample = results_df.iloc[indices]
        routed = sample[sample["confidence"] >= threshold]
        coverage = len(routed) / n
        boot_covs.append(coverage)
        if len(routed) > 0:
            boot_accs.append(routed["correct"].mean())
        else:
            boot_accs.append(0.0)

    alpha = (1.0 - ci) / 2.0
    acc_ci = np.percentile(boot_accs, [alpha * 100, (1 - alpha) * 100])
    cov_ci = np.percentile(boot_covs, [alpha * 100, (1 - alpha) * 100])
    return acc_ci.tolist(), cov_ci.tolist()
