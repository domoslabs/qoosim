import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_simulation_results(input_folder, output_file="simulation_plots.png"):
    """Plot the sojourn times from multiple simulation results as time-series with enhanced readability for A4 output.

    Parameters:
        input_folder (str): Path to the folder containing simulation results.
        output_file (str): Path to save the resulting plot.
    """
    result_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".npy")]
    result_files.sort()

    # Adjust figure size for A4 paper when reduced
    num_plots = len(result_files)
    fig_height = 3 * (num_plots - 1)  # Increased height to fit larger text
    fig_width = 20  # this will scale down but keep aspect ratio

    fig, axes = plt.subplots(num_plots, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [85, 15]}, sharex='col', sharey='col')
    if num_plots == 1:
        axes = axes.reshape(1, -1)

    # Add title to the figure
    fig.suptitle("WiFi Simulation Results: Sojourn Times and CDFs", fontsize=20)  # y adjusts spacing from top

    for i, result_file in enumerate(result_files):
        results = np.load(result_file, allow_pickle=True)
        sojourn_times = np.array([departure - arrival for arrival, departure, _ in results])
        time_stamps = np.array([departure for arrival, departure, _ in results])

        # Time-series plot
        axes[i, 0].plot(time_stamps, sojourn_times, color="blue")
        axes[i, 0].set_ylabel("Sojourn Time (s)", fontsize=14)  # Increased font size
        axes[i, 0].grid(True, linestyle='--', alpha=0.7)
        axes[i, 0].set_title(f"Run {i+1}", fontsize=16)  # Increased font size

        # CDF plot
        sorted_times = np.sort(sojourn_times)
        ecdf = np.arange(1, len(sojourn_times)+1) / (len(sojourn_times)+1)
        axes[i, 1].plot(sorted_times, ecdf, color='green')
        axes[i, 1].set_xlabel('Sojourn Time (s)', fontsize=12)  # Increased font size
        axes[i, 1].set_ylabel('CDF', fontsize=12)  # Increased font size
        axes[i, 1].grid(True, linestyle='--', alpha=0.7)
        axes[i, 1].set_title("CDF", fontsize=14)  # Increased font size
        axes[i, 1].tick_params(axis='both', which='major', labelsize=10)  # Increased tick label size

    # Common labels
    fig.text(0.5, 0.04, "Time (s)", ha="center", va="center", fontsize=16)  # Larger font for x-label

    # Finalize plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout to make room for the title
    plt.rcParams['figure.dpi'] = 300  # Higher DPI for better quality when resized
    plot_dir = os.path.join(input_folder, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, output_file), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    experiment = "wifi" # "wifi", "bufferbloat_1sec", "bufferbloat_5sec" or "service_outage"
    input_folder = f"{experiment}_simulation_results"
    plot_simulation_results(input_folder=input_folder, output_file=f"{experiment}_simulation_time_series_cdf_readable.png")