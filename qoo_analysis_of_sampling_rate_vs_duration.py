import os
import numpy as np
import matplotlib.pyplot as plt
from simulator import generate_latency_trace
from simulator import compute_qoo_score, sub_sample_trace_exponential_index_space

def run_qoo_stability_analysis(input_file, nrp, nrpou, original_frequency=1000, sub_frequencies=[10, 50, 100], iterations=10):
    """Run QoO stability analysis on a single simulation result for different time segments.

    Parameters:
        input_file (str): Path to the simulation result file.
        nrp (dict): Network Requirements for Perfection.
        nrpou (dict): Network Requirement Point of Unusableness.
        original_frequency (int): Original sampling frequency in Hz.
        sub_frequencies (list): List of sub-sampling frequencies to simulate.
        iterations (int): Number of iterations for each frequency.
    """
    
    results = np.load(input_file, allow_pickle=True)
    simulation_time = 5000 # Fixed simulation time for this example
    latency_trace = generate_latency_trace(results, simulation_time=simulation_time, sampling_frequency=original_frequency)
    ground_truth = latency_trace[:, 1]

    segments = [300, 600, 1200, 2400, 4800]

    stability_results = {}
    ground_truth_qoos = {}

    for segment in segments:
        segment_trace = ground_truth[:segment * original_frequency]  # Slice the trace up to the segment
        ground_truth_qoo = compute_qoo_score(segment_trace, packet_loss=0, nrp=nrp, nrpou=nrpou)
        
        stability_scores = {}
        for sub_frequency in sub_frequencies:
            scores = []
            for _ in range(iterations):
                sampled_trace = sub_sample_trace_exponential_index_space(segment_trace, original_frequency=original_frequency, sub_frequency=sub_frequency)
                qoo_score = compute_qoo_score(sampled_trace, packet_loss=0, nrp=nrp, nrpou=nrpou)
                scores.append(qoo_score)
            stability_scores[sub_frequency] = scores

        stability_results[segment] = stability_scores
        ground_truth_qoos[segment] = ground_truth_qoo

    return stability_results, ground_truth_qoos

def plot_results(input_file, stability_results, ground_truth_qoos, sub_frequencies):
    """Plot QoO stability results for different segments of the trace."""
    fig, axes = plt.subplots(len(stability_results), 1, figsize=(15, 3 * len(stability_results)), sharex=True, sharey=True)
    if len(stability_results) == 1:
        axes = [axes]  # Ensure axes is iterable even for one plot

    for ax, (segment, stability_scores) in zip(axes, stability_results.items()):
        ax.set_title(f"QoO Stability for first {segment} seconds", fontsize=12, pad=10)

        data = [stability_scores[sub_frequency] for sub_frequency in sub_frequencies]
        ax.boxplot(data, labels=sub_frequencies, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="blue"),
                   medianprops=dict(color="red"))
        ax.axhline(y=ground_truth_qoos[segment], color="green", linestyle="--", label="Ground Truth QoO")

        ax.set_ylabel("QoO Score")
        ax.set_ylim(-1, 101)
        ax.grid(True)

    axes[-1].set_xlabel("Sub-Sampling Frequency (Hz)")
    fig.suptitle(f"QoO Stability Analysis for {os.path.basename(input_file)}", fontsize=16, y=0.98)
    plt.subplots_adjust(hspace=0.7)

    # Ensure the plots folder exists
    plot_folder = os.path.join(os.path.dirname(input_file), "plots")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
        
    plt.savefig(os.path.join(plot_folder, f"qoo_stability_analysis_{os.path.basename(input_file)}.png"))
    plt.close(fig)  # Close the figure to free up memory and avoid display for each plot

if __name__ == "__main__":
    input_folder = "wifi_simulation_results_longer_duration"

    nrp = {95.0: 0.100, 99.0: 0.200, 'loss': 0.1}
    nrpou = {95.0: 0.400, 99.0: 0.450, 'loss': 1.0}
    sub_frequencies = [0.05, 0.1, 0.2, 1, 5, 10, 50, 100]
    iterations = 100

    # Get all result files
    result_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".npy")]

    for result_file in result_files:
        stability_results, ground_truth_qoos = run_qoo_stability_analysis(
            result_file, nrp, nrpou, original_frequency=1000, sub_frequencies=sub_frequencies, iterations=iterations
        )
        plot_results(result_file, stability_results, ground_truth_qoos, sub_frequencies)