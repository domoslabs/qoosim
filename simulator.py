import numpy as np
import matplotlib.pyplot as plt
import heapq

# Discrete event simulator with a queue
def discrete_event_simulator(packet_generators, inter_packet_delays, simulation_time):
    """
    Simulates a queue using discrete events.

    Parameters:
        packet_generators (list): List of packet generation functions, each returning inter-arrival times.
        inter_packet_delays (list): List of delays between packet departures.
        simulation_time (float): Total simulation time in seconds.

    Returns:
        list: Arrival and departure times of packets.
    """
    # Priority queue for events
    event_queue = []

    clock = 0
    # Initialize packet generation events
    for generator_id, generator in enumerate(packet_generators):
        next_time = generator(clock)
        heapq.heappush(event_queue, (next_time, 'arrival', generator_id))

    # Simulate the queue
    queue = []
    next_departure = 0
    results = []

    while event_queue and clock < simulation_time:
        # Pop the next event
        clock, event_type, source_id = heapq.heappop(event_queue)

        if event_type == 'arrival':
            # Handle packet arrival
            queue.append(clock)
            next_time = clock + packet_generators[source_id](clock)
            if next_time < simulation_time:
                heapq.heappush(event_queue, (next_time, 'arrival', source_id))

        elif event_type == 'departure':
            # Handle packet departure
            if queue:
                arrival_time = queue.pop(0)
                departure_time = clock
                results.append((arrival_time, departure_time, len(queue)))

        # Schedule the next departure if there are packets in the queue
        if queue:
            next_departure = max(next_departure, clock) + np.random.choice(inter_packet_delays)
            heapq.heappush(event_queue, (next_departure, 'departure', None))

    return results

# Example packet generator: Poisson process
def poisson_packet_generator(rate):
    """
    Returns a function that generates inter-arrival times based on a Poisson process.

    Parameters:
        rate (float): Rate (packets per second).

    Returns:
        function: A function generating inter-arrival times.
    """
    def generator(clock):
        return np.random.exponential(1.0 / rate)

    return generator

def isochronous_packet_generator(rate):
    """ 
    Returns a function that generates inter-arrival times based on an isochronous process.
    
    Parameters:
        rate (float): Rate (packets per second).
        
    Returns:
        function: A function generating inter-arrival times.
    """
    def generator(clock):
        return 1.0 / rate
    
    return generator

def bursty_packet_generator(rate, burst_interval):
    """
    Returns a function that generates inter-arrival times based on a bursty process.

    Parameters:
        rate (float): Rate (packets per second).
        burst_interval (float): Time between bursts.

    Returns:
        function: A function generating inter-arrival times.
    """
    def generator(clock):
        if int(clock) % burst_interval == 0:
            return np.random.exponential(1.0 / rate)
        else:
            return 1
    return generator

def bursty_packet_generator_long_bursts(rate, burst_interval, burst_length):
    """
    Returns a function that generates inter-arrival times based on a bursty process.

    Parameters:
        rate (float): Rate (packets per second).
        burst_interval (float): Time between bursts.

    Returns:
        function: A function generating inter-arrival times.
    """
    def generator(clock):
        if int(clock) % burst_interval < burst_length:
            return np.random.exponential(1.0 / rate)
        else:
            return 1
    return generator

# Generate a latency trace by interpolating sojourn times
def generate_latency_trace(results, simulation_time, sampling_frequency):
    """
    Generate a latency trace sampled at a specific frequency by interpolating sojourn times.

    Parameters:
        results (list): List of tuples (arrival_time, departure_time, queue_depth) from the simulation.
        simulation_time (float): Total simulation time in seconds.
        sampling_frequency (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Time-stamped latency trace sampled at the specified frequency.
    """
    sample_times = np.arange(0, simulation_time, 1 / sampling_frequency)
    
    sojourn_times = []
    departure_times = []
    for arrival, departure, queue_depth in results:
        sojourn_times.append(departure - arrival)
        departure_times.append(departure)
    
    # Interpolate sojourn times
    sojourn_trace = np.interp(sample_times, departure_times, sojourn_times)
    return np.column_stack((sample_times, sojourn_trace))


# Generate a synthetic ground truth latency trace using the simulator
def generate_ground_truth(packet_generators, inter_packet_delays, duration=10, frequency=1000):
    """
    Generates a ground truth latency trace.

    Parameters:
        duration (int): Duration of the trace in seconds.
        frequency (int): Sampling frequency in Hz.

    Returns:
        np.ndarray: Ground truth latency trace.
    """
    results = discrete_event_simulator(packet_generators, inter_packet_delays, duration)

    # Generate the latency trace by interpolating sojourn times
    latency_trace = generate_latency_trace(results, duration, frequency)
    return latency_trace[:, 0], latency_trace[:, 1], results
    

def sub_sample_trace_exponential_index_space(ground_truth, original_frequency, sub_frequency):
    """
    Sub-sample a latency trace with exponentially distributed inter-arrival indices.
    On average, sub_frequency 'samples per second' translates to
    mean index increment = original_frequency / sub_frequency.
    """
    mean_interval = original_frequency / sub_frequency  # in "index units"
    sample_indices = [0]  # always start with index 0
    current_index = 0.0

    while True:
        # Add a continuous exponential increment in "index" space
        current_index += np.random.exponential(mean_interval)
        # Round or floor to get an actual integer index
        next_idx = int(round(current_index))
        if next_idx < len(ground_truth):
            sample_indices.append(next_idx)
        else:
            break

    sampled_trace = ground_truth[sample_indices]
    return sampled_trace

# Compute QoO score based on specification
def compute_qoo_score(latency_trace, packet_loss, nrp, nrpou):
    """
    Compute the QoO score based on latency trace and packet loss.

    Parameters:
        latency_trace (np.ndarray): Latency trace.
        packet_loss (float): Measured packet loss (percentage).
        nrp (dict): Network Requirements for Perfection.
        nrpou (dict): Network Requirement Point of Unusableness.

    Returns:
        float: QoO score.
    """
    # Calculate percentiles for the latency trace
    measured_percentiles = {
        p: np.percentile(latency_trace, p) for p in nrp.keys() if p != 'loss'
    }

    # Compute latency-based QoO score
    qoo_latency_scores = []
    for percentile, nrp_value in nrp.items():
        if percentile == 'loss':
            continue
        nrpou_value = nrpou[percentile]
        measured_value = measured_percentiles[percentile]
        qoo_latency = max(
            0, min((1 - ((measured_value - nrp_value) / (nrpou_value - nrp_value))) * 100, 100)
        )
        qoo_latency_scores.append(qoo_latency)

    qoo_latency = min(qoo_latency_scores)

    # Compute packet-loss-based QoO score
    nrp_loss = nrp.get('loss', 0)
    nrpou_loss = nrpou.get('loss', 100)
    qoo_loss = max(
        0, min((1 - ((packet_loss - nrp_loss) / (nrpou_loss - nrp_loss))) * 100, 100)
    )

    # Final QoO score
    qoo_score = min(qoo_latency, qoo_loss)
    return qoo_score

# Visualize ground truth sojourn time and queue depth
def plot_ground_truth(results, duration, sampling_frequency, ax1, ax2):
    """
    Plot the ground truth sojourn time and queue depth.

    Parameters:
        results (list): Simulation results containing arrival and departure times and queue depth.
        duration (int): Duration of the simulation.
        sampling_frequency (int): Sampling frequency for the plot.
        ax1 (matplotlib.axes.Axes): Axis for sojourn time.
        ax2 (matplotlib.axes.Axes): Axis for queue depth.
    """
    # Generate time points for sampling
    sample_times = np.arange(0, duration, 1 / sampling_frequency)

    # Interpolate sojourn times and queue depths
    sojourn_times = []
    queue_depths = []
    for t in sample_times:
        # Find the last recorded event before or at time t
        last_event = next(((arr, dep, depth) for arr, dep, depth in reversed(results) if dep <= t), None)
        if last_event:
            arrival_time, departure_time, queue_depth = last_event
            sojourn_times.append(departure_time - arrival_time)
            queue_depths.append(queue_depth)
        else:
            sojourn_times.append(0)
            queue_depths.append(0)

    # Plot sojourn times and queue depth
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Sojourn Time (s)", color="tab:blue")
    ax1.plot(sample_times, sojourn_times, label="Sojourn Time", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2.set_ylabel("Queue Depth", color="tab:orange")
    ax2.plot(sample_times, queue_depths, label="Queue Depth", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

# Simulate QoO scores for different sampling frequencies and accuracies
def simulate_qoo_scores(packet_generators, inter_packet_delays, duration=10, original_frequency=1000, sub_frequencies=[1, 10, 100], iterations=10, nrp=None, nrpou=None):
    """
    Simulate QoO scores for different sub-sampling frequencies and noise levels.

    Parameters:
        duration (int): Duration of the trace in seconds.
        original_frequency (int): Original sampling frequency in Hz.
        sub_frequencies (list): List of sub-sampling frequencies to simulate.
        iterations (int): Number of iterations for each frequency.
        nrp (dict): Network Requirements for Perfection.
        nrpou (dict): Network Requirement Point of Unusableness.

    Returns:
        dict: Simulated QoO scores.
    """
    if nrp is None:
        nrp = {99: 0.250, 99.9: 0.350, 'loss': 0.1}
    if nrpou is None:
        nrpou = {99: 0.400, 99.9: 0.450, 'loss': 1.0}

    time_points, ground_truth, results = generate_ground_truth(packet_generators, inter_packet_delays, duration, original_frequency)
    results_data = {}

    # Compute ground truth QoO
    ground_truth_packet_loss = 0  # Assume no packet loss for ground truth
    ground_truth_qoo = compute_qoo_score(ground_truth, ground_truth_packet_loss, nrp, nrpou)

    for sub_frequency in sub_frequencies:
        scores = []
        for _ in range(iterations):
            sampled_trace = sub_sample_trace_exponential_index_space(ground_truth, original_frequency, sub_frequency)
            packet_loss = 0 
            qoo_score = compute_qoo_score(sampled_trace, packet_loss, nrp, nrpou)
            scores.append(qoo_score)
        results_data[sub_frequency] = scores

    return results_data, ground_truth_qoo, results

# Wrapper function to plot both ground truth and QoO scores
def plot_combined(results_data, ground_truth_qoo, simulation_results, duration, original_frequency):
    """
    Plot ground truth (sojourn time and queue depth) and QoO scores in the same figure.

    Parameters:
        results_data (dict): Simulated QoO scores.
        ground_truth_qoo (float): Ground truth QoO score.
        simulation_results (list): Simulation results for ground truth.
        duration (int): Duration of the simulation.
        original_frequency (int): Original sampling frequency.
    """
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot QoO results
    frequencies = sorted(results_data.keys())
    data = [results_data[freq] for freq in frequencies]

    ax3.boxplot(data, positions=frequencies, widths=5)
    ax3.axhline(y=ground_truth_qoo, color='r', linestyle='--', label="Ground Truth QoO")
    ax3.set_xlabel("Sampling Frequency (Hz)")
    ax3.set_ylabel("QoO Score")
    ax3.set_title("Effect of Sampling Frequency on QoO Score Variability")
    ax3.legend()
    ax3.grid(True)

    # Plot ground truth
    ax2 = ax1.twinx()
    plot_ground_truth(simulation_results, duration, original_frequency, ax1, ax2)

    fig.tight_layout()
    plt.show()