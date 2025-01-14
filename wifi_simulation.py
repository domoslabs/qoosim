import os
import numpy as np
from simulator import discrete_event_simulator, poisson_packet_generator

def run_wifi_simulation(output_folder, iterations=10):
    """Run WiFi environment simulation and save results."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    duration = 300  # seconds

    # WiFi environment configuration
    packet_generators = [
        poisson_packet_generator(170),  # Background traffic
    ]

    # Inter-packet delays for WiFi scenario
    inter_packet_delays = np.genfromtxt('reference_data/1_competing_sta.csv', delimiter=',')
    inter_packet_delays = inter_packet_delays[~np.isnan(inter_packet_delays)]
    inter_packet_delays = inter_packet_delays * 0.001 * 0.1666  # Convert to seconds and correct for the large TxOP of the competing traffic in source data (12ms)

    for i in range(iterations):
        results = discrete_event_simulator(packet_generators, inter_packet_delays, duration)
        output_path = os.path.join(output_folder, f"wifi_simulation_run_{i+1}.npy")
        np.save(output_path, results)
        print(f"Saved WiFi simulation run {i+1} to {output_path}")

if __name__ == "__main__":
    run_wifi_simulation(output_folder="wifi_simulation_results")
