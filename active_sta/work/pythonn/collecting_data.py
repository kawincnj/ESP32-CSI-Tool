import serial
import time
import re
import csv
import os
import math
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PORT       = '/dev/ttyUSB0'
BAUDRATE   = 921600

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SAVE_DIR   = os.path.join(PROJECT_ROOT, 'csi_dataset')

# ↓↓ CHANGE THESE BEFORE EACH SESSION ↓↓
LABEL      = 'standing'   # sitting / standing / walking / bending / lying
SUBJECT_ID = 'subject_01'

# Filter out bad packets
MIN_CSI_LEN = 64   # minimum number of subcarriers expected

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
filename     = os.path.join(SAVE_DIR, f'{LABEL}_{SUBJECT_ID}_{session_time}.csv')

ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=1)
time.sleep(2)

packet_count  = 0
skip_count    = 0
start_time    = time.time()

print(f"[INFO] Label     : {LABEL}")
print(f"[INFO] Subject   : {SUBJECT_ID}")
print(f"[INFO] Saving to : {filename}")
print(f"[INFO] Min CSI   : {MIN_CSI_LEN} subcarriers")
print(f"[INFO] Press Ctrl+C to stop\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_amplitudes(line):
    """Extract and compute amplitudes from CSI line. Returns list or None."""

    # Grab content inside LAST [...] in the line (most reliable)
    matches = re.findall(r'\[([^\[\]]*)\]', line)
    if not matches:
        return None

    raw_str  = matches[-1]  # last bracket = CSI buffer
    raw_vals = [int(x) for x in raw_str.split() if x]

    if len(raw_vals) < MIN_CSI_LEN * 2:  # *2 because I+Q pairs
        return None

    imag = raw_vals[0::2]
    real = raw_vals[1::2]
    n    = min(len(imag), len(real))

    amplitudes = [
        round(math.sqrt(imag[i]**2 + real[i]**2), 4)
        for i in range(n)
    ]

    # Per-packet normalization to 0–1
    max_val    = max(amplitudes) if max(amplitudes) > 0 else 1
    normalized = [round(a / max_val, 6) for a in amplitudes]

    return amplitudes, normalized


# ── Main ──────────────────────────────────────────────────────────────────────
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        'timestamp',
        'label',
        'subject_id',
        'rssi',
        'noise_floor',
        'channel',
        'packet_num',
        'csi_len',
        'csi_amplitudes',   # raw — for inspection
        'csi_normalized'    # normalized — use THIS for LSTM input
    ])

    try:
        while True:
            if ser.in_waiting == 0:
                continue

            raw  = ser.readline()
            line = raw.decode('utf-8', errors='replace').rstrip()

            if 'CSI_DATA' not in line:
                continue

            # Parse fields
            fields = line.split(',')
            try:
                rssi        = int(fields[3])
                noise_floor = int(fields[13])
                channel     = int(fields[15])
            except (IndexError, ValueError):
                skip_count += 1
                continue

            # Parse CSI
            result = parse_amplitudes(line)
            if result is None:
                skip_count += 1
                continue

            amplitudes, normalized = result

            # Write
            packet_count += 1
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                LABEL,
                SUBJECT_ID,
                rssi,
                noise_floor,
                channel,
                packet_count,
                len(amplitudes),
                amplitudes,
                normalized
            ])
            csvfile.flush()

            # Terminal feedback every 10 packets
            if packet_count % 10 == 0:
                elapsed = time.time() - start_time
                rate    = packet_count / elapsed if elapsed > 0 else 0
                print(f"[{packet_count:>5}] RSSI:{rssi:>4} dBm | "
                      f"CSI:{len(amplitudes):>3} subcarriers | "
                      f"{rate:.1f} pkt/s | "
                      f"Skipped:{skip_count} | "
                      f"{elapsed:.1f}s elapsed")

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n{'─'*50}")
        print(f"[DONE] Label    : {LABEL}")
        print(f"[DONE] Packets  : {packet_count} saved, {skip_count} skipped")
        print(f"[DONE] Duration : {elapsed:.1f} seconds")
        print(f"[DONE] Rate     : {packet_count/elapsed:.1f} pkt/s")
        print(f"[DONE] File     : {filename}")
        print(f"{'─'*50}")
    finally:
        ser.close()
