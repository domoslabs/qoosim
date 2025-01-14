import os
import numpy as np
from simulator import discrete_event_simulator, poisson_packet_generator

def run_service_outage_simulation(output_folder, iterations=10):
    """Run service outage simulation and save results."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    duration = 300  # seconds

    # Service outage configuration
    packet_generators = [
        poisson_packet_generator(100),  # Background traffic
    ]

    # Inter-packet delays simulating service outage
    inter_packet_delays = [0.0025] * 6000 + [1.0]  # Simulate a service outage with large delays

    for i in range(iterations):
        results = discrete_event_simulator(packet_generators, inter_packet_delays, duration)
        output_path = os.path.join(output_folder, f"service_outage_simulation_run_{i+1}.npy")
        np.save(output_path, results)
        print(f"Saved service outage simulation run {i+1} to {output_path}")

if __name__ == "__main__":
    run_service_outage_simulation(output_folder="service_outage_simulation_results")
