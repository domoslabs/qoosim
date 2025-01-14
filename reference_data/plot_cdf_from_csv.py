import numpy as np
import matplotlib.pyplot as plt
plt.figure()

for nsta in [1]:
    inter_packet_delays = np.genfromtxt(f'{nsta}_competing_sta.csv', delimiter=',')
    inter_packet_delays = inter_packet_delays[~np.isnan(inter_packet_delays)]
    inter_packet_delays = inter_packet_delays * 0.001 * 0.1666 
    inter_packet_delays = inter_packet_delays.tolist()

    cdf = np.cumsum(inter_packet_delays)
    cdf = cdf / cdf[-1]

    # Plot the CDF of the inter-packet delays
    plt.plot(np.sort(inter_packet_delays), cdf, label=f'{nsta} competing STAs')
plt.xlabel('Inter-packet delay (s)')
plt.ylabel('CDF')
plt.title('CDF of Inter-packet Delays for a mildly congested WiFi network')
plt.grid(True)
plt.legend()
plt.savefig('wifi_cdf_1_sta.png', dpi=300)
plt.show()
