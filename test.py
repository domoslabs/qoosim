import unittest
import numpy as np
from unittest.mock import patch

from simulator import (
    compute_qoo_score, 
    generate_latency_trace, 
    poisson_packet_generator,
    isochronous_packet_generator,
    bursty_packet_generator, 
    bursty_packet_generator_long_bursts, 
    discrete_event_simulator, 
    sub_sample_trace_exponential_index_space,
    generate_ground_truth,
    simulate_qoo_scores
)

class TestDiscreteEventSimulator(unittest.TestCase):

    def setUp(self):
        # Mock packet generators for deterministic testing
        self.fixed_gen = lambda clock: 1.0  # Always return 1 second as inter-arrival time
        self.inter_packet_delays = [1.0]

    def test_discrete_event_simulator(self):
        results = discrete_event_simulator([self.fixed_gen], self.inter_packet_delays, simulation_time=3.0)
        # Expect 2 packets in 3 seconds with 1 second fixed delay
        # Results on format [(arrival_time, departure_time, queue_depth), ...]
        expected = [(1.0, 2.0, 1), (2.0, 3.0, 0.0)]
        self.assertEqual(len(results), len(expected))
        for i, (arrival, departure, queue_depth) in enumerate(results):
            self.assertAlmostEqual(arrival, expected[i][0], places=7)
            self.assertAlmostEqual(departure, expected[i][1], places=7)
            self.assertEqual(queue_depth, expected[i][2])

    def test_generate_latency_trace(self):
        # Test interpolation of latency trace
        results = [(1.0, 2.0, 0), (2.0, 3.0, 0)]  # Simulated result from above
        trace = generate_latency_trace(results, simulation_time=3.0, sampling_frequency=2)
        # With sampling_frequency=2, we expect sample times: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        expected_trace = np.array([
            [0.0,  1.0],
            [0.5,  1.0],
            [1.0,  1.0],
            [1.5,  1.0],
            [2.0,  1.0],
            [2.5,  1.0]
        ])
        np.testing.assert_array_almost_equal(trace, expected_trace, decimal=7)

    def test_compute_qoo_score(self):
        nrp = {99: 0.1, 99.9: 0.14}
        nrpou = {99: 0.3, 99.9: 0.4}
        latency_trace = np.array([0.15] * 1000)  # All latencies are exactly 0.15s
        qoo_score = compute_qoo_score(latency_trace, packet_loss=0.0, nrp=nrp, nrpou=nrpou)
        expected_score = 75.0
        self.assertAlmostEqual(qoo_score, expected_score, places=7)

    @patch('numpy.random.exponential')
    def test_bursty_packet_generator(self, mock_exp):
        """
        Test that bursty_packet_generator returns short inter-arrivals during 'bursts'
        and 1 second otherwise, for a given burst_interval.
        """
        mock_exp.return_value = 0.5  # fix the exponential to always return 0.5
        burst_interval = 2
        rate = 2.0  # 2 packets/s
        gen = bursty_packet_generator(rate, burst_interval)

        # For times that are multiples of burst_interval, the generator should return 0.5
        # else it should return 1.0
        self.assertAlmostEqual(gen(0), 0.5)   # 0 % 2 == 0 => burst
        self.assertAlmostEqual(gen(1), 1.0)   # 1 % 2 != 0 => not burst
        self.assertAlmostEqual(gen(2), 0.5)   # 2 % 2 == 0 => burst
        self.assertAlmostEqual(gen(2.1), 0.5) # int(2.1) % 2 == 0 => burst

    @patch('numpy.random.exponential')
    def test_bursty_packet_generator_long_bursts(self, mock_exp):
        """
        Test that bursty_packet_generator_long_bursts returns short inter-arrivals 
        for a continuous 'burst_length' within the 'burst_interval'.
        """
        mock_exp.return_value = 0.3
        rate = 2.0
        burst_interval = 5
        burst_length = 2
        gen = bursty_packet_generator_long_bursts(rate, burst_interval, burst_length)

        # For times 0 <= t < 2 (i.e., from 0 to <2 within each 0..5 block), it should return 0.3
        # Otherwise 1.0
        self.assertAlmostEqual(gen(0), 0.3)  # 0 % 5 = 0 < 2 => burst
        self.assertAlmostEqual(gen(1.9), 0.3)  # 1.9 % 5 = 1.9 < 2 => burst
        self.assertAlmostEqual(gen(2.0), 1.0)  # 2.0 % 5 = 2 => outside the burst window
        self.assertAlmostEqual(gen(6.9), 0.3)  # (6.9 % 5) = 1.9 => within the burst
        self.assertAlmostEqual(gen(7.0), 1.0)  # (7 % 5) = 2 => outside the burst window

    def test_isochronous_packet_generator(self):
        """
        Test that isochronous_packet_generator returns a constant inter-arrival time of 1/rate.
        """
        rate = 10.0  # 10 packets/s => inter-arrival is 0.1
        gen = isochronous_packet_generator(rate)
        self.assertAlmostEqual(gen(0), 0.1)
        self.assertAlmostEqual(gen(10), 0.1)
        self.assertAlmostEqual(gen(100), 0.1)

    @patch('numpy.random.poisson')
    def test_sub_sample_trace_poisson_no_noise(self, mock_poisson):
        """
        Test sub_sample_trace_poisson with a fixed Poisson sequence (no noise).
        """
        # Force a deterministic Poisson output
        # For example, always return intervals of 2
        mock_poisson.return_value = 2

        ground_truth = np.linspace(0, 1, 10)  # 10 data points, e.g., [0, 0.111..., 0.222..., 1.0]
        original_freq = 100
        sub_freq = 50

        # This means the mean interval = 100 / 50 = 2
        # So the index increments by 2 each time.
        # We'll keep going until current_time >= len(ground_truth).
        # So we sample indices: 0, 2, 4, 6, 8
        sampled = sub_sample_trace_exponential_index_space(ground_truth, original_freq, sub_freq, noise_std=0)

        # Verify the sampled trace is correct
        expected = ground_truth[[0, 2, 4, 6, 8]]
        np.testing.assert_array_almost_equal(sampled, expected, decimal=7)

    def test_generate_ground_truth(self):
        """
        Test that generate_ground_truth returns time_points, latencies, and raw results
        with expected shapes.
        """
        # Use a simple single fixed generator for reproducibility
        time_points, latencies, results = generate_ground_truth(
            packet_generators=[self.fixed_gen],
            inter_packet_delays=self.inter_packet_delays,
            duration=5,
            frequency=10
        )
        # Expect length of time_points and latencies = duration * frequency = 50
        self.assertEqual(len(time_points), 50)
        self.assertEqual(len(latencies), 50)
        # Check that results is a list of (arrival, departure, queue_depth)
        self.assertTrue(all(len(r) == 3 for r in results))

    def test_simulate_qoo_scores(self):
        """
        Test simulate_qoo_scores ensures correct structure and that ground_truth_qoo 
        is within a valid range [0..100].
        """
        packet_generators = [self.fixed_gen]
        sub_frequencies = [10, 20]
        noise_stds = [0, 5]
        results_data, ground_truth_qoo, simulation_results = simulate_qoo_scores(
            packet_generators,
            self.inter_packet_delays,
            duration=3,
            original_frequency=100,
            sub_frequencies=sub_frequencies,
            noise_stds=noise_stds,
            iterations=2
        )
        # Check keys in results_data
        self.assertCountEqual(results_data.keys(), sub_frequencies)
        # Each key should map to a list of length = iterations
        for freq in sub_frequencies:
            self.assertEqual(len(results_data[freq]), 2)
        # ground_truth_qoo should be between 0 and 100
        self.assertGreaterEqual(ground_truth_qoo, 0)
        self.assertLessEqual(ground_truth_qoo, 100)
        # simulation_results is the discrete-event simulation results
        self.assertTrue(all(len(r) == 3 for r in simulation_results))

if __name__ == '__main__':
    unittest.main()
