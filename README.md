# qoosim
Simulations of Quality of Outcome in various network scenarios  


The repository consists of 13 python files.

simulator.py contains the code for simulating the FIFO queue, as well as the code for the packet generators.

These files produce simulation results for each of the scenarios described in the Method section:
* bufferbloat_1sec_simulation.py
* bufferbloat_5sec_simulation.py
* wifi_simulation.py
* service_outage_simulation.py
* wifi_simulation_longer_duration.py

These files produce plots. To produce plots for a specific scenario, change the value of the “experiment” variable.
* qoo_analysis_of_measurement_accuracy.py
  * Adds noise to the simulated results
* qoo_analysis_of_requirement_sensitivity.py
  * Tests different application requirements 
* qoo_analysis_of_sampling_rate.py
  * Tests different sampling rates
* qoo_analysis_of_sampling_rate_vs_duration.py
  * The analysis tests different measurement durations. Designed to work only on the wifi_simulation_longer_duration results.
* plot_sojourns.py
  * Plots the simulated ground-truths from an experiment as time-series and CDFs

run_all_analysis.py will produce all of the plots except the ones made by qoo_analysis_of_sampling_rate_vs_duration.py

test.py contains a few simple tests of the simulator and QoO computations, and can also serve as examples for how to use the different functions.
