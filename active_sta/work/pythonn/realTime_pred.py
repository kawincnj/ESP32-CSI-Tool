import serial
import time
import re
import math
import collections
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
PORT        = 'COM7'
BAUDRATE    = 921600
MODEL_PATH  = 'models/best_posture_lstm.pth'
DATA_DIR    = 'prepared_data'
WINDOW_SIZE = 100
MIN_CSI_LEN = 64

# How many consecutive same predictions before announcing
CONFIRM_THRESHOLD = 3

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] {device}")

# ── Load normalization stats ──────────────────────────────────────────────────
global_mean = np.load(f'{DATA_DIR}/global_mean.npy')
global_std  = np.load(f'{DATA_DIR}/global_std.npy')
classes     = np.load(f'{DATA_DIR}/classes.npy')
NUM_CLASSES = len(classes)
print(f"[INFO] Classes: {list(classes)}")

# ── Model definition (must match train_lstm.py) ───────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = 64,
            hidden_size   = 128,
            num_layers    = 2,
            batch_first   = True,
            dropout       = 0.3,
            bidirectional = True
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]
        return self.classifier(out)

# ── Load model ────────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = PostureLSTM().to(device)
checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"[INFO] Model loaded — trained to epoch {checkpoint['epoch']} "
      f"| best val acc: {checkpoint['val_acc']:.1f}%\n")

# ── CSI parser ────────────────────────────────────────────────────────────────
def parse_amplitudes(line):
    matches = re.findall(r'\[([^\[\]]*)\]', line)
    if not matches:
        return None

    raw_vals = [int(x) for x in matches[-1].split() if x]
    if len(raw_vals) < MIN_CSI_LEN * 2:
        return None

    imag = raw_vals[0::2]
    real = raw_vals[1::2]
    n    = min(len(imag), len(real))

    amplitudes = np.array([
        math.sqrt(imag[i]**2 + real[i]**2)
        for i in range(n)
    ], dtype=np.float32)[:64]

    if len(amplitudes) < 64:
        return None

    # Apply same global normalization as training
    normalized = (amplitudes - global_mean) / global_std
    return normalized.astype(np.float32)

# ── Predict from window ───────────────────────────────────────────────────────
def predict(window):
    """window: list of 100 arrays each shape (64,)"""
    x = np.stack(window)                          # (100, 64)
    x = torch.tensor(x).unsqueeze(0).to(device)   # (1, 100, 64)

    with torch.no_grad():
        logits = model(x)                          # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)      # (1, num_classes)

    probs_np    = probs.squeeze().cpu().numpy()    # (num_classes,)
    pred_idx    = int(probs_np.argmax())
    pred_label  = classes[pred_idx]
    confidence  = float(probs_np[pred_idx]) * 100

    return pred_label, confidence, probs_np

# ── Display ───────────────────────────────────────────────────────────────────
POSTURE_ART = {
    'sitting' : '🪑 SITTING',
    'standing': '🧍 STANDING',
    'walking' : '🚶 WALKING',
    'bending' : '🙇 BENDING',
    'lying'   : '🛌 LYING',
}

def display(label, confidence, probs, confirmed):
    art   = POSTURE_ART.get(label, f'❓ {label.upper()}')
    bar   = '█' * int(confidence / 5) + '░' * (20 - int(confidence / 5))
    tag   = '✅ CONFIRMED' if confirmed else '   ...'

    print(f"\n{'─'*50}")
    print(f"  {art}  {tag}")
    print(f"  Confidence: [{bar}] {confidence:.1f}%")
    print(f"  All classes:")
    for i, cls in enumerate(classes):
        p   = probs[i] * 100
        b   = '█' * int(p / 5)
        print(f"    {cls:>12} : {b:<20} {p:.1f}%")
    print(f"  Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"{'─'*50}")

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"[INFO] Opening {PORT} @ {BAUDRATE}...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Connected! Filling window ({WINDOW_SIZE} packets)...\n")
except Exception as e:
    print(f"[ERROR] Cannot open port: {e}")
    exit(1)

# Sliding window buffer
buffer           = collections.deque(maxlen=WINDOW_SIZE)
packet_count     = 0
last_prediction  = None
confirm_count    = 0

try:
    while True:
        if ser.in_waiting == 0:
            continue

        raw  = ser.readline()
        line = raw.decode('utf-8', errors='replace').rstrip()

        if 'CSI_DATA' not in line:
            continue

        amp = parse_amplitudes(line)
        if amp is None:
            continue

        buffer.append(amp)
        packet_count += 1

        # Show filling progress
        if packet_count <= WINDOW_SIZE and packet_count % 10 == 0:
            print(f"  Filling buffer: {packet_count}/{WINDOW_SIZE} packets...")

        # Only predict when buffer is full
        if len(buffer) < WINDOW_SIZE:
            continue

        # Predict every 10 new packets (sliding window)
        if packet_count % 10 != 0:
            continue

        label, confidence, probs = predict(list(buffer))

        # Confirmation logic — same prediction N times = confirmed
        if label == last_prediction:
            confirm_count += 1
        else:
            confirm_count   = 1
            last_prediction = label

        confirmed = confirm_count >= CONFIRM_THRESHOLD

        display(label, confidence, probs, confirmed)

except KeyboardInterrupt:
    print(f"\n[DONE] Stopped. Total packets processed: {packet_count}")
finally:
    ser.close()