import os
import math
import numpy as np
import matplotlib.pyplot as plt
from simulator import generate_latency_trace
from simulator import compute_qoo_score, sub_sample_trace_exponential_index_space
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from simulator import generate_latency_trace
from simulator import compute_qoo_score, sub_sample_trace_exponential_index_space

def run_qoo_one_pair_per_row_hybrid(
    input_folder,
    nrp_nrpou_pairs,
    original_frequency=1000,
    sub_frequency=50,
    iterations=10,
    output_file="qoo_one_pair_per_row_hybrid.png",
    figure_title="Latency Requirement & QoO Comparisons",
    simulation_time=900
):
    """
    Figures produced:

    1) Figure 1 (pairs-based):
       - One row per (NRP, NRPOU) pair.
       - Left: Requirements, Right: Boxplot across ALL runs.

    2) Figure 2 (runs-based, 2 columns):
       - Up to 2 runs per row.
       - Each subplot: boxplot comparing all pairs on that run.

    3) Figure 3 (first-run-only):
       - n_rows = n_pairs, 2 columns in total.
       - Left column: one subplot per pair (requirements).
       - Right column: one big subplot that spans all rows (boxplot for run #0
         with one box per pair).
    """

    # 1) Gather runs
    result_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".npy")
    )
    n_runs = len(result_files)
    n_pairs = len(nrp_nrpou_pairs)

    # 2) Prepare data structure: qoo_scores[pair_idx][run_idx] => list of QoO
    qoo_scores = [ [ [] for _ in range(n_runs) ] for _ in range(n_pairs) ]

    # 3) Compute QoO
    for run_idx, run_file in enumerate(result_files):
        results = np.load(run_file, allow_pickle=True)
        latency_trace = generate_latency_trace(
            results, simulation_time=simulation_time, sampling_frequency=original_frequency
        )
        ground_truth = latency_trace[:, 1]

        for pair_idx, (nrp_dict, nrpou_dict) in enumerate(nrp_nrpou_pairs):
            for _ in range(iterations):
                sampled = sub_sample_trace_exponential_index_space(
                    ground_truth,
                    original_frequency=original_frequency,
                    sub_frequency=sub_frequency
                )
                score = compute_qoo_score(
                    sampled, packet_loss=0,
                    nrp=nrp_dict,
                    nrpou=nrpou_dict
                )
                qoo_scores[pair_idx][run_idx].append(score)

    # -------------------------------
    # 4) Global axis limits
    # -------------------------------
    # A) Requirements x-limits
    all_req_vals = []
    for (nrp_dict, nrpou_dict) in nrp_nrpou_pairs:
        for d in (nrp_dict, nrpou_dict):
            for v in d.values():
                if isinstance(v, (int, float)):
                    all_req_vals.append(v)

    if all_req_vals:
        req_min = min(all_req_vals)
        req_max = max(all_req_vals)
        margin = 0.05 * (req_max - req_min)
        global_min_req_x = req_min - margin
        global_max_req_x = req_max + margin
    else:
        global_min_req_x, global_max_req_x = 0, 1

    # B) QoO y-limits
    all_qoo = []
    for pair_idx in range(n_pairs):
        for run_idx in range(n_runs):
            all_qoo.extend(qoo_scores[pair_idx][run_idx])

    if all_qoo:
        qoo_min = min(all_qoo)
        qoo_max = max(all_qoo)
        margin = 0.05 * (qoo_max - qoo_min)
        global_min_qoo_y = qoo_min - margin
        global_max_qoo_y = qoo_max + margin
    else:
        global_min_qoo_y, global_max_qoo_y = 0, 1

    # ---------------------------------------------------------
    # Figure 1 (pairs-based, each row => left: req, right: QoO)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(
        nrows=n_pairs, ncols=2,
        figsize=(14, 3 * n_pairs)
    )
    if n_pairs == 1:
        axes1 = [axes1]

    for pair_idx, (nrp_dict, nrpou_dict) in enumerate(nrp_nrpou_pairs):
        ax_req = axes1[pair_idx][0]
        ax_box = axes1[pair_idx][1]

        # (A) Left: Requirements
        ax_req.set_title(f"Requirement R{pair_idx+1}", fontsize=10)
        ax_req.set_xlim(global_min_req_x, global_max_req_x)
        ax_req.grid(True)

        # gather percentile keys
        pct_keys = sorted(k for k in set(nrp_dict.keys()).union(nrpou_dict.keys())
                          if isinstance(k, float))
        y_positions = list(range(len(pct_keys) + 1))  # +1 for 'loss'
        y_labels = [f"{p}%" for p in pct_keys]
        y_loss = len(pct_keys)
        y_labels.append("loss")

        ax_req.set_yticks(y_positions)
        ax_req.set_yticklabels(y_labels)

        # Draw lines from NRP -> NRPOU
        for i, p in enumerate(pct_keys):
            if p in nrp_dict and p in nrpou_dict:
                ax_req.plot([nrp_dict[p], nrpou_dict[p]], [i, i], 'b-', linewidth=3)
                ax_req.plot(nrp_dict[p], i, 'bo')
                ax_req.plot(nrpou_dict[p], i, 'bx')

        if 'loss' in nrp_dict and 'loss' in nrpou_dict:
            ax_req.plot([nrp_dict['loss'], nrpou_dict['loss']], [y_loss, y_loss],
                        'r-', linewidth=3)
            ax_req.plot(nrp_dict['loss'], y_loss, 'ro')
            ax_req.plot(nrpou_dict['loss'], y_loss, 'rx')

        # (B) Right: Boxplot across ALL runs
        ax_box.set_title(f"R{pair_idx+1}: QoO (All Runs)", fontsize=10)
        ax_box.set_ylim(global_min_qoo_y, global_max_qoo_y)
        ax_box.set_ylabel("QoO Score")
        ax_box.grid(True)

        # each run => one box
        data_for_boxplot = [qoo_scores[pair_idx][r] for r in range(n_runs)]
        run_labels = [f"Run {i+1}" for i in range(n_runs)]
        ax_box.boxplot(
            data_for_boxplot,
            labels=run_labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red")
        )
        ax_box.set_xticklabels(run_labels, rotation=25)

    fig1.suptitle(figure_title, fontsize=14)
    plt.tight_layout()
    fig1.subplots_adjust(hspace=0.5)
    if not os.path.exists(f"{input_folder}/plots"):
        os.makedirs(f"{input_folder}/plots")
    plt.savefig(f"{input_folder}/plots/{output_file}")
    # plt.show()
    plt.close(fig1)  # Close the figure to free up memory
    # -------------------------------------------------
    # Figure 2 (runs-based, 2 columns)
    # -------------------------------------------------
    nrow = math.ceil(n_runs / 2)
    ncol = 2

    fig2, axes2 = plt.subplots(
        nrows=nrow, ncols=ncol,
        figsize=(8.27, 11.69),
        sharey=True
    )
    axes2 = np.atleast_2d(axes2)
    pair_labels = [f"R{i+1}" for i in range(n_pairs)]

    for i in range(n_runs):
        rr = i // ncol
        cc = i % ncol
        ax2 = axes2[rr, cc]
        ax2.set_title(f"Run {i+1} QoO Scores by Requirement", fontsize=9)
        ax2.set_ylim(global_min_qoo_y, global_max_qoo_y)
        ax2.grid(True)

        # one box per pair
        data_run = []
        for pair_idx in range(n_pairs):
            data_run.append(qoo_scores[pair_idx][i])
        ax2.boxplot(
            data_run,
            labels=pair_labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red")
        )
        ax2.set_xticklabels(pair_labels, rotation=25)
        ax2.set_ylabel("QoO")

    # remove extra subplots if any
    total_subplots = nrow * ncol
    for idx in range(n_runs, total_subplots):
        rr = idx // ncol
        cc = idx % ncol
        fig2.delaxes(axes2[rr, cc])

    fig2.suptitle(figure_title, fontsize=12)
    plt.tight_layout()
    fig2.subplots_adjust(hspace=0.6, wspace=0.3)
    runs_output = output_file.replace(".png", "_all_runs.png")
    if not os.path.exists(f"{input_folder}/plots"):
        os.makedirs(f"{input_folder}/plots")
    plt.savefig(f"{input_folder}/plots/{runs_output}")
    # plt.show()
    plt.close(fig2)  # Close the figure to free up memory
    # -------------------------------------------------
    # Figure 3 (single-run scenario: n_rows = n_pairs, 1 big boxplot for run #0)
    # -------------------------------------------------
    # Condition: if we have at least 1 run
    if n_runs > 0:
        fig3 = plt.figure(figsize=(14, 3*n_pairs))

        # We'll create a GridSpec so that the left col has n_pairs subplots,
        # and the right col is 1 big subplot spanning all rows.
        gs = fig3.add_gridspec(nrows=n_pairs, ncols=2, width_ratios=[3, 1])

        left_axes = []
        # Each row => separate subplot for that pair's requirements
        for i in range(n_pairs):
            ax_req = fig3.add_subplot(gs[i, 0])
            left_axes.append(ax_req)

        # Single big subplot for the right column
        ax_box_single = fig3.add_subplot(gs[:, 1])

        # (A) Fill the left column: each pair's requirement plot
        for pair_idx, (nrp_dict, nrpou_dict) in enumerate(nrp_nrpou_pairs):
            ax_req = left_axes[pair_idx]
            ax_req.set_title(f"Requirement R{pair_idx+1}", fontsize=10)
            ax_req.set_xlim(global_min_req_x, global_max_req_x)
            ax_req.grid(True)

            pct_keys = sorted(k for k in set(nrp_dict.keys()).union(nrpou_dict.keys())
                              if isinstance(k, float))
            y_positions = list(range(len(pct_keys) + 1))
            y_labels = [f"{p}%" for p in pct_keys]
            y_loss = len(pct_keys)
            y_labels.append("loss")

            ax_req.set_yticks(y_positions)
            ax_req.set_yticklabels(y_labels)

            # lines
            for i, p in enumerate(pct_keys):
                if p in nrp_dict and p in nrpou_dict:
                    ax_req.plot([nrp_dict[p], nrpou_dict[p]], [i, i],
                                'b-', linewidth=3)
                    ax_req.plot(nrp_dict[p], i, 'bo')
                    ax_req.plot(nrpou_dict[p], i, 'bx')
            if 'loss' in nrp_dict and 'loss' in nrpou_dict:
                ax_req.plot([nrp_dict['loss'], nrpou_dict['loss']],
                            [y_loss, y_loss], 'r-', linewidth=3)
                ax_req.plot(nrp_dict['loss'], y_loss, 'ro')
                ax_req.plot(nrpou_dict['loss'], y_loss, 'rx')

        # (B) Fill the single big subplot on the right with one run's boxplot
        # We combine all pairs => one box per pair
        ax_box_single.set_title("First Run Only: QoO (All Requirements)", fontsize=10)
        ax_box_single.set_ylim(global_min_qoo_y, global_max_qoo_y)
        ax_box_single.set_ylabel("QoO Score")
        ax_box_single.grid(True)

        # build data: one box per pair from run #0
        data_first_run = []
        pair_labels_first_run = []
        for pair_idx in range(n_pairs):
            data_first_run.append(qoo_scores[pair_idx][0])  # run_idx=0
            pair_labels_first_run.append(f"R{pair_idx+1}")

        ax_box_single.boxplot(
            data_first_run,
            labels=pair_labels_first_run,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            medianprops=dict(color="red")
        )
        ax_box_single.set_xticklabels(pair_labels_first_run, rotation=25)
        left_axes[-1].set_xlabel("Seconds")

        fig3.suptitle(f"{figure_title} - Single Run 1 Data", fontsize=13)
        plt.tight_layout()
        # A bit more space between subplots
        fig3.subplots_adjust(hspace=0.7, wspace=0.4)

        first_run_output = output_file.replace(".png", "_first_run.png")
        if not os.path.exists(f"{input_folder}/plots"):
            os.makedirs(f"{input_folder}/plots")
        plt.savefig(f"{input_folder}/plots/{first_run_output}")
        #plt.show()
        plt.close(fig3)  # Close the figure to free up memory



# ------------------------------------------------------------------
# Example usage: run the function for each set of requirement pairs
# ------------------------------------------------------------------
if __name__ == "__main__":
    experiment = "wifi" # "wifi", "bufferbloat_1sec", "bufferbloat_5sec" or "service_outage"
    input_folder = f"{experiment}_simulation_results"

    nrp_nrpou_pairs_nrpou_fixed = [
        # 1) Pair A
        (
            {90.0: 0.1, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            {90.0: 0.11, 95.0: 0.16, 99.0: 0.21},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            {90.0: 0.12, 95.0: 0.17, 99.0: 0.22},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            {90.0: 0.13, 95.0: 0.18, 99.0: 0.23},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            {90.0: 0.14, 95.0: 0.19, 99.0: 0.24},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
    ]

    nrp_nrpou_pairs_nrp_fixed = [
        (
            # Pair 1: baseline NRPOU
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            # Pair 2: slight increase
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.36, 95.0: 0.41, 99.0: 0.46}
        ),
        (
            # Pair 3
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.37, 95.0: 0.42, 99.0: 0.47}
        ),
        (
            # Pair 4
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.38, 95.0: 0.43, 99.0: 0.48}
        ),
        (
            # Pair 5
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.39, 95.0: 0.44, 99.0: 0.49}
        ),
    ]

    nrp_nrpou_pairs_shifted_percentiles = [
        (
            # Pair 1 (lower percentiles)
            {89.0: 0.10, 94.0: 0.15, 98.0: 0.20},
            {89.0: 0.35, 94.0: 0.40, 98.0: 0.45}
        ),
        (
            # Pair 2 (original-ish: 90, 95, 99)
            {90.0: 0.10, 95.0: 0.15, 99.0: 0.20},
            {90.0: 0.35, 95.0: 0.40, 99.0: 0.45}
        ),
        (
            # Pair 3
            {91.0: 0.10, 96.0: 0.15, 99.5: 0.20},
            {91.0: 0.35, 96.0: 0.40, 99.5: 0.45}
        ),
        (
            # Pair 4
            {92.0: 0.10, 97.0: 0.15, 99.5: 0.20},
            {92.0: 0.35, 97.0: 0.40, 99.5: 0.45}
        ),
        (
            # Pair 5 (highest percentiles)
            {93.0: 0.10, 98.0: 0.15, 99.9: 0.20},
            {93.0: 0.35, 98.0: 0.40, 99.9: 0.45}
        ),
    ]

    # -- 1) Gradually Increasing NRP (NRPOU fixed)
    run_qoo_one_pair_per_row_hybrid(
        input_folder=input_folder,
        nrp_nrpou_pairs=nrp_nrpou_pairs_nrpou_fixed,
        original_frequency=1000,
        sub_frequency=10,
        iterations=100,
        output_file=f"qoo_nrp_var_{experiment}.png",
        figure_title="Gradually Changing NRP (NRPOU Fixed)",
        simulation_time=300
    )

    # -- 2) Gradually Increasing NRPOU (NRP fixed)
    run_qoo_one_pair_per_row_hybrid(
        input_folder=input_folder,
        nrp_nrpou_pairs=nrp_nrpou_pairs_nrp_fixed,
        original_frequency=1000,
        sub_frequency=10,
        iterations=100,
        output_file=f"qoo_nrpou_var_{experiment}.png",
        figure_title="Gradually Changing NRPOU (NRP Fixed)",
        simulation_time=300
    )

    # -- 3) Shifted Percentiles
    run_qoo_one_pair_per_row_hybrid(
        input_folder=input_folder,
        nrp_nrpou_pairs=nrp_nrpou_pairs_shifted_percentiles,
        original_frequency=1000,
        sub_frequency=10,
        iterations=100,
        output_file=f"qoo_shifted_pct_{experiment}.png",
        figure_title="Shifting Percentile Definitions",
        simulation_time=300
    )
