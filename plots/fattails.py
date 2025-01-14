import numpy as np
import os
import matplotlib.pyplot as plt

def sample_fat_tailed_distribution(alpha=1.5, size=10000, scale=1.0):
    """
    Generate samples from a Pareto (fat-tailed) distribution.
    The Pareto PDF is: f(x) = alpha * scale^alpha / x^(alpha+1), for x >= scale.

    Parameters:
        alpha (float): shape parameter (alpha < 2 => heavy tails).
        size (int): number of samples.
        scale (float): minimum (scale) parameter for the Pareto distribution.

    Returns:
        np.ndarray: array of samples from Pareto.
    """
    # Using NumPy’s Pareto draws: X ~ Pareto(alpha) => actual sample = scale*(X+1).
    raw = np.random.pareto(alpha, size=size)
    return scale * (raw + 1.0)

def theoretical_pareto_percentile(alpha, scale=1.0, p=0.99):
    """
    Theoretical p-th percentile (quantile) of Pareto, using the inverse CDF:
      Q(p) = scale * (1 - p)^(-1/alpha),  for 0 < p < 1.
    If alpha <= 0 or p <= 0 or p>=1, we return None.

    E.g., for p=0.99:
      Q(0.99) = scale * (0.01)^(-1/alpha).
    """
    if alpha <= 0 or p <= 0 or p >= 1:
        return None
    return scale * ((1.0 - p) ** (-1.0 / alpha))

def demonstrate_99th_percentile_estimation(
    alpha=1.5,
    scale=1.0,
    sample_sizes=None,
    n_experiments=500,
    percentile=99
):
    """
    Illustrate how small samples from a fat-tailed (Pareto) distribution
    can yield misleading or highly variable estimates of an extreme percentile 
    (e.g., the 99th percentile).

    Steps:
      1) Generate a moderate sample (10k) for log-scale histogram visualization.
      2) For each sample size in `sample_sizes`, repeat n_experiments times:
         - Draw a sample and estimate its 99th percentile via np.percentile.
      3) Plot a boxplot of these percentile estimates.
      4) Add a horizontal line for the theoretical 99th percentile if alpha>0.
    """

    if sample_sizes is None:
        sample_sizes = [10, 100, 1000]

    # 1) Moderate sample (for histogram)
    moderate_data = sample_fat_tailed_distribution(alpha=alpha, size=10000, scale=scale)

    # 2) Collect repeated percentile estimates
    percentile_estimates_by_size = []
    for sz in sample_sizes:
        estimates = []
        for _ in range(n_experiments):
            data = sample_fat_tailed_distribution(alpha=alpha, size=sz, scale=scale)
            # estimate the empirical 99th percentile
            est_pctl = np.percentile(data, percentile)
            estimates.append(est_pctl)
        percentile_estimates_by_size.append(estimates)

    # 3) Theoretical 99th percentile
    true_99th = theoretical_pareto_percentile(alpha=alpha, scale=scale, p=percentile/100.0)

    # 4) Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (A) Log-scale histogram of the moderate sample
    axes[0].hist(moderate_data, bins=80, color="skyblue", edgecolor="k", log=True)
    axes[0].set_title(f"Pareto Dist (alpha={alpha}, scale={scale})\nLog-scale Histogram")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency (log scale)")

    # (B) Boxplot of 99th-percentile estimates
    box_labels = [str(sz) for sz in sample_sizes]
    box = axes[1].boxplot(percentile_estimates_by_size, labels=box_labels, patch_artist=True)

    # Color boxplot
    for patch in box['boxes']:
        patch.set_facecolor("lightcoral")

    axes[1].set_title(f"{percentile}th Percentile Estimates\n({n_experiments} experiments/sample size)")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel(f"Estimated {percentile}th Percentile")

    # Horizontal line for the theoretical value
    if true_99th is not None:
        axes[1].axhline(y=true_99th, color='blue', linestyle='--',
                        label=f"True {percentile}th = {true_99th:.2f}")
        axes[1].legend()

    fig.suptitle("Limitations of Small Samples for Fat-Tailed Distributions", fontsize=14)
    plt.tight_layout()
    # Check that the plots folder exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/pareto_percentile_estimation.png")
    plt.show()

if __name__ == "__main__":
    """
    Example usage:
      - We'll fix a random seed for reproducibility.
      - We'll illustrate alpha=1.5, which is fairly heavy-tailed but still < 2.
    """
    np.random.seed(43)
    demonstrate_99th_percentile_estimation(
        alpha=1.5, 
        scale=1.0, 
        sample_sizes=[10, 100, 1000], 
        n_experiments=500,
        percentile=99
    )
