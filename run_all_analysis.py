from plot_sojourns import plot_simulation_results
from qoo_analysis_of_sampling_rate import run_qoo_stability_analysis
from qoo_analysis_of_measurement_accuracy import run_qoo_measurement_accuracy_stability_analysis
from qoo_analysis_of_requirement_sensitivity import run_qoo_one_pair_per_row_hybrid

for experiment in ["bufferbloat_1sec", "bufferbloat_5sec", "service_outage", "wifi"]:
    input_folder = f"{experiment}_simulation_results"
    
    nrp = {95.0: 0.100, 99.0: 0.200, 'loss': 0.1}
    nrpou = {95.0: 0.400, 99.0: 0.450, 'loss': 1.0}
    
    plot_simulation_results(input_folder=input_folder, output_file=f"{experiment}_simulation_time_series_cdf_readable.png")

    run_qoo_stability_analysis(
        input_folder=input_folder,
        nrp=nrp,
        nrpou=nrpou,
        original_frequency=1000,
        sub_frequencies=[0.05, 0.1, 0.2, 1, 5, 10, 50, 100],
        iterations=100,
        output_file=f"qoo_stability_analysis_{experiment}.png"
    )
    
    run_qoo_measurement_accuracy_stability_analysis(
        input_folder=input_folder,
        nrp=nrp,
        nrpou=nrpou,
        original_frequency=1000,
        iterations=100,
        noise_levels=[0.01, 0.05, 0.1],
        output_file=f"qoo_stability_analysis_{experiment}_measurement_noise.png"
    )
    
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
        figure_title="Gradually Changing NRP (NRPOU Fixed)"
    )

    # -- 2) Gradually Increasing NRPOU (NRP fixed)
    run_qoo_one_pair_per_row_hybrid(
        input_folder=input_folder,
        nrp_nrpou_pairs=nrp_nrpou_pairs_nrp_fixed,
        original_frequency=1000,
        sub_frequency=10,
        iterations=100,
        output_file=f"qoo_nrpou_var_{experiment}.png",
        figure_title="Gradually Changing NRPOU (NRP Fixed)"
    )

    # -- 3) Shifted Percentiles
    run_qoo_one_pair_per_row_hybrid(
        input_folder=input_folder,
        nrp_nrpou_pairs=nrp_nrpou_pairs_shifted_percentiles,
        original_frequency=1000,
        sub_frequency=10,
        iterations=100,
        output_file=f"qoo_shifted_pct_{experiment}.png",
        figure_title="Shifting Percentile Definitions"
    )
