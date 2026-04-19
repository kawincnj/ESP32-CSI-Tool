import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_utils'))

import serial
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
import collections
from wait_timer import WaitTimer

parser = argparse.ArgumentParser()
parser.add_argument('--port',  default='COM7', help='Serial port')
parser.add_argument('--baud',  default=921600, type=int)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW       = 100   # packets to show
SMOOTH       = 5     # moving average window
MOTION_THRESHOLD = 8 # tune this to your environment

print_stats_wait_timer = WaitTimer(1.0)
render_plot_wait_timer = WaitTimer(0.1)  # faster refresh

perm_amp   = collections.deque(maxlen=WINDOW)
motion_energy = collections.deque(maxlen=WINDOW)

packet_count        = 0
total_packet_counts = 0
prev_mean_amp       = None

# ── Plot setup: 3 panels ──────────────────────────────────────────────────────
plt.ion()
fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

ax1 = fig.add_subplot(gs[0])  # Heatmap
ax2 = fig.add_subplot(gs[1])  # Mean amplitude + smoothed
ax3 = fig.add_subplot(gs[2])  # Motion energy

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460')

plt.show(block=False)

# ── Helpers ───────────────────────────────────────────────────────────────────
def moving_average(data, n):
    if len(data) < n:
        return list(data)
    return list(np.convolve(data, np.ones(n)/n, mode='valid'))

def plot_all(amp, energy):
    # --- Panel 1: Heatmap of all subcarriers ---
    ax1.cla()
    arr = np.array(list(amp))          # shape: (time, subcarriers)
    ax1.imshow(arr.T, aspect='auto', cmap='plasma',
               origin='lower', interpolation='nearest')
    ax1.set_title("All Subcarriers Heatmap", fontsize=10, pad=4)
    ax1.set_xlabel("Time (packets)")
    ax1.set_ylabel("Subcarrier")
    ax1.set_facecolor('#16213e')

    # --- Panel 2: Mean amplitude across all subcarriers ---
    ax2.cla()
    mean_amps = [np.mean(row) for row in amp]
    smoothed  = moving_average(mean_amps, SMOOTH)
    x_raw     = list(range(len(mean_amps)))
    x_smooth  = list(range(len(mean_amps) - len(smoothed), len(mean_amps)))
    ax2.plot(x_raw,    mean_amps, color='#0f9b8e', alpha=0.4, linewidth=1,   label='Raw')
    ax2.plot(x_smooth, smoothed,  color='#00d4ff', linewidth=2, label='Smoothed')
    ax2.set_title("Mean Amplitude (all subcarriers)", fontsize=10, pad=4)
    ax2.set_xlabel("Time (packets)")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(0, WINDOW)
    ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax2.set_facecolor('#16213e')

    # --- Panel 3: Motion energy (diff between consecutive frames) ---
    ax3.cla()
    energy_list = list(energy)
    colors = ['#ff4d4d' if e > MOTION_THRESHOLD else '#4dff91' for e in energy_list]
    x_e = list(range(len(energy_list)))
    ax3.bar(x_e, energy_list, color=colors, width=1.0)
    ax3.axhline(y=MOTION_THRESHOLD, color='yellow', linewidth=1,
                linestyle='--', label=f'Threshold ({MOTION_THRESHOLD})')
    ax3.set_title("Motion Energy  🔴 = Movement Detected  🟢 = Static", fontsize=10, pad=4)
    ax3.set_xlabel("Time (packets)")
    ax3.set_ylabel("Energy")
    ax3.set_xlim(0, WINDOW)
    ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax3.set_facecolor('#16213e')

    fig.canvas.flush_events()
    plt.show()

# ── CSI parser ────────────────────────────────────────────────────────────────
def process(line):
    global prev_mean_amp
    all_data = line.split(',')

    csi_raw = None
    for field in all_data:
        if '[' in field:
            csi_raw = field
            break
    if csi_raw is None:
        return

    csi_raw  = csi_raw.replace('[', '').replace(']', '').strip()
    csi_data = [int(x) for x in csi_raw.split() if x]
    if len(csi_data) < 4:
        return

    imaginary = csi_data[0::2]
    real      = csi_data[1::2]

    amplitudes = [
        math.sqrt(imaginary[i]**2 + real[i]**2)
        for i in range(min(len(imaginary), len(real)))
    ]

    if not amplitudes:
        return

    perm_amp.append(amplitudes)

    # Compute motion energy = mean absolute diff from previous frame
    cur_mean = np.mean(amplitudes)
    if prev_mean_amp is not None:
        energy = abs(cur_mean - prev_mean_amp)
        motion_energy.append(energy)
    prev_mean_amp = cur_mean

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"[CSI] Opening {args.port} @ {args.baud}...")
try:
    ser = serial.Serial(args.port, args.baud, timeout=1)
    print("[CSI] Connected! Waiting for data...\n")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

while True:
    try:
        raw  = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', errors='replace').strip()

        if "CSI_DATA" not in line:
            continue

        process(line)
        packet_count        += 1
        total_packet_counts += 1

        if print_stats_wait_timer.check():
            print_stats_wait_timer.update()
            print(f"[STATS] {packet_count} pkt/s | Total: {total_packet_counts}")
            packet_count = 0

        if render_plot_wait_timer.check() and len(perm_amp) > 5:
            render_plot_wait_timer.update()
            plot_all(perm_amp, motion_energy)

    except KeyboardInterrupt:
        print("\n[CSI] Stopped.")
        ser.close()
        break
    except Exception as e:
        print(f"[ERROR] {e}")
        continue

'''
python serial_plot_csi_live.py --port COM7 --baud 921600
'''
