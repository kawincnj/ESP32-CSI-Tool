import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import collections
from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line

# Set subcarrier to plot
subcarrier = 20

# Wait Timers. Change these values to increase or decrease the rate of `print_stats` and `render_plot`.
print_stats_wait_timer = WaitTimer(1.0)
render_plot_wait_timer = WaitTimer(0.2)

# Deque definition
perm_amp = collections.deque(maxlen=100)
perm_phase = collections.deque(maxlen=100)

# Variables to store CSI statistics
packet_count = 0
total_packet_counts = 0

# Create figure for plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
fig.canvas.draw()
plt.show(block=False)


def carrier_plot(amp):
    plt.clf()
    df = np.asarray(amp, dtype=object)
    
    # Find shortest row to avoid index errors
    min_len = min(len(row) for row in amp)
    if subcarrier >= min_len:
        print(f"Subcarrier {subcarrier} out of range (max {min_len-1})")
        return

    df_vals = [row[subcarrier] for row in amp]
    plt.plot(range(100 - len(amp), 100), df_vals, color='r')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.xlim(0, 100)
    plt.title(f"Amplitude plot of Subcarrier {subcarrier}")
    fig.canvas.flush_events()
    plt.show()


def process(res):
    # Split CSV fields
    all_data = res.split(',')

    # Find the field that starts with '[' (CSI buffer)
    csi_raw = None
    for field in all_data:
        if '[' in field:
            csi_raw = field
            break

    if csi_raw is None:
        return

    # Strip brackets and split by whitespace (space-separated)
    csi_raw = csi_raw.replace('[', '').replace(']', '').strip()
    csi_data = [int(x) for x in csi_raw.split() if x]

    if len(csi_data) < 2:
        return

    # Split into imaginary and real (alternating: imag, real, imag, real...)
    imaginary = csi_data[0::2]
    real      = csi_data[1::2]

    amplitudes = []
    phases = []
    for i in range(min(len(imaginary), len(real))):
        amp   = math.sqrt(imaginary[i] ** 2 + real[i] ** 2)
        phase = math.atan2(imaginary[i], real[i])
        amplitudes.append(amp)
        phases.append(phase)

    if amplitudes:
        perm_amp.append(amplitudes)
        perm_phase.append(phases)

print_until_first_csi_line()

while True:
    line = readline()
    if "CSI_DATA" in line:
        process(line)
        packet_count += 1
        total_packet_counts += 1

        if print_stats_wait_timer.check():
            print_stats_wait_timer.update()
            print("Packet Count:", packet_count, "per second.", "Total Count:", total_packet_counts)
            packet_count = 0

        if render_plot_wait_timer.check() and len(perm_amp) > 2:
            render_plot_wait_timer.update()
            carrier_plot(perm_amp)
