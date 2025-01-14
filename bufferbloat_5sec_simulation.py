import os
import numpy as np
from simulator import discrete_event_simulator, poisson_packet_generator, bursty_packet_generator_long_bursts

def run_bufferbloat_simulation(output_folder, iterations=10):
    """Run bufferbloat environment simulation and save results."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    duration = 300  # seconds

    # Bufferbloat environment configuration
    packet_generators = [
        poisson_packet_generator(200),  # Background traffic
        bursty_packet_generator_long_bursts(220, 30, 5),  # Bursty traffic for the bufferbloat scenario
    ]

    # Inter-packet delays simulating bufferbloat
    inter_packet_delays = [0.0025]

    for i in range(iterations):
        results = discrete_event_simulator(packet_generators, inter_packet_delays, duration)
        output_path = os.path.join(output_folder, f"bufferbloat_5sec_simulation_run_{i+1}.npy")
        np.save(output_path, results)
        print(f"Saved bufferbloat simulation run {i+1} to {output_path}")

if __name__ == "__main__":
    run_bufferbloat_simulation(output_folder="bufferbloat_5sec_simulation_results")
