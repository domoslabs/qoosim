import os
import numpy as np
import matplotlib.pyplot as plt
from simulator import generate_latency_trace
from simulator import compute_qoo_score, sub_sample_trace_exponential_index_space

def run_qoo_stability_analysis(input_folder, nrp, nrpou, original_frequency=1000, sub_frequencies=[10, 50, 100], iterations=10, output_file="qoo_stability_analysis.png"):
    """Run QoO stability analysis on multiple simulation results.

    Parameters:
        input_folder (str): Path to the folder containing simulation results.
        nrp (dict): Network Requirements for Perfection.
        nrpou (dict): Network Requirement Point of Unusableness.
        original_frequency (int): Original sampling frequency in Hz.
        sub_frequencies (list): List of sub-sampling frequencies to simulate.
        iterations (int): Number of iterations for each frequency.
        output_file (str): Path to save the resulting plot.
    """
    # Get all result files
    result_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".npy")]
    result_files.sort()  # Ensure consistent ordering

    stability_results = {}
    ground_truth_qoos = {}

    # Analyze each result file
    for result_file in result_files:    
        results = np.load(result_file, allow_pickle=True)
        latency_trace = generate_latency_trace(results, simulation_time=300, sampling_frequency=original_frequency)
        ground_truth = latency_trace[:, 1]
        ground_truth_qoo = compute_qoo_score(ground_truth, packet_loss=0, nrp=nrp, nrpou=nrpou)
        stability_scores = {}

        for sub_frequency in sub_frequencies:
            scores = []
            for _ in range(iterations):
                sampled_trace = sub_sample_trace_exponential_index_space(ground_truth, original_frequency=1000, sub_frequency=sub_frequency)
                qoo_score = compute_qoo_score(sampled_trace, packet_loss=0, nrp=nrp, nrpou=nrpou)
                scores.append(qoo_score)
            stability_scores[sub_frequency] = scores

        stability_results[result_file] = stability_scores
        ground_truth_qoos[result_file] = ground_truth_qoo

 
    # Plot results
    num_files = len(result_files)
    fig, axes = plt.subplots(int(np.ceil(num_files / 2)), 2, figsize=(15, 3 * int(np.ceil(num_files / 2))), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (result_file, stability_scores) in enumerate(stability_results.items()):
        ax = axes[i]
        ax.set_title(f"QoO Stability for Run {i + 1}", fontsize=12, pad=10)

        data = [stability_scores[sub_frequency] for sub_frequency in sub_frequencies]
        ax.boxplot(data, labels=sub_frequencies, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="blue"),
                   medianprops=dict(color="red"))
        ax.axhline(y=ground_truth_qoos[result_file], color="green", linestyle="--", label="Ground Truth QoO")

        ax.set_ylabel("QoO Score")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    axes[-2].set_xlabel("Sub-Sampling Frequency (Hz)")
    axes[-1].set_xlabel("Sub-Sampling Frequency (Hz)")
    fig.suptitle("QoO Stability Analysis (Boxplots)", fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.7)
    # Check that the plots folder exists
    if not os.path.exists(f"{input_folder}/plots"):
        os.makedirs(f"{input_folder}/plots")
    plt.savefig(f"{input_folder}/plots/{output_file}")
    #plt.show()
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    # Example usage
    experiment = "wifi" # "bufferbloat" "service_outage" or "wifi" 
    input_folder = f"{experiment}_simulation_results"  # Change this to the desired simulation folder

    nrp = {95.0: 0.100, 99.0: 0.200, 'loss': 0.1}
    nrpou = {95.0: 0.400, 99.0: 0.450, 'loss': 1.0}

    run_qoo_stability_analysis(
        input_folder=input_folder,
        nrp=nrp,
        nrpou=nrpou,
        original_frequency=1000,
        sub_frequencies=[0.05, 0.1, 0.2, 1, 5, 10, 50, 100],
        iterations=100,
        output_file=f"qoo_stability_analysis_{experiment}.png"
    )
