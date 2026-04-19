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
MODEL_PATH  = 'models/best_cnn_lstm_attn.pth'
DATA_DIR    = 'prepared_data'
WINDOW_SIZE = 100
MIN_CSI_LEN = 64
CONFIRM_N   = 3

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] {device}\n")

# ── Load stats ────────────────────────────────────────────────────────────────
global_mean = np.load(f'{DATA_DIR}/global_mean.npy')
global_std  = np.load(f'{DATA_DIR}/global_std.npy')
classes     = np.load(f'{DATA_DIR}/classes.npy')
NUM_CLASSES = len(classes)

# ── Model definition ──────────────────────────────────────────────────────────
# Must match train_cnn_lstm.py exactly
CNN_CHANNELS = [32, 64, 128]
KERNEL_SIZE  = 3
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
ATTN_HEADS   = 4
DROPOUT      = 0.3
INPUT_SIZE   = 64

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        out, w = self.attn(x, x, x)
        return self.norm(x + out), w

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv    = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.ca      = ChannelAttention(out_ch)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        return self.dropout(self.ca(self.conv(x)))

class CNNLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_layers = []
        in_ch = INPUT_SIZE
        for out_ch in CNN_CHANNELS:
            cnn_layers.append(CNNBlock(in_ch, out_ch, KERNEL_SIZE))
            in_ch = out_ch
        self.cnn  = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(CNN_CHANNELS[-1], LSTM_HIDDEN, LSTM_LAYERS,
                            batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.attn = TemporalAttention(LSTM_HIDDEN * 2, ATTN_HEADS)
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64),              nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attn(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ── Load model ────────────────────────────────────────────────────────────────
print("[INFO] Loading model...")
model = CNNLSTMAttention().to(device)
ckpt  = torch.load(MODEL_PATH, weights_only=False, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"[INFO] Loaded — epoch {ckpt['epoch']} | best val acc: {ckpt['val_acc']:.1f}%")
print(f"[INFO] Classes: {list(classes)}\n")

# ── CSI parser ────────────────────────────────────────────────────────────────
def parse_amplitudes(line):
    matches = re.findall(r'\[([^\[\]]*)\]', line)
    if not matches:
        return None
    raw = [int(x) for x in matches[-1].split() if x]
    if len(raw) < MIN_CSI_LEN * 2:
        return None
    imag = raw[0::2]
    real = raw[1::2]
    n    = min(len(imag), len(real))
    amp  = np.array([math.sqrt(imag[i]**2 + real[i]**2)
                     for i in range(n)], dtype=np.float32)[:64]
    if len(amp) < 64:
        return None
    return ((amp - global_mean) / global_std).astype(np.float32)

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(window):
    x      = np.stack(window)                          # (100, 64)
    x      = torch.tensor(x).unsqueeze(0).to(device)  # (1, 100, 64)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx    = int(probs.argmax())
    return classes[idx], float(probs[idx]) * 100, probs

# ── Display ───────────────────────────────────────────────────────────────────
ICONS = {
    'sitting' : '🪑', 'standing': '🧍',
    'walking' : '🚶', 'bending' : '🙇', 'lying': '🛌'
}

def display(label, conf, probs, confirmed):
    icon  = ICONS.get(label, '❓')
    bar   = '█' * int(conf / 5) + '░' * (20 - int(conf / 5))
    tag   = '✅ CONFIRMED' if confirmed else '🔄 detecting...'
    print(f"\n{'─'*52}")
    print(f"  {icon} {label.upper():<12}  {tag}")
    print(f"  Confidence : [{bar}] {conf:.1f}%")
    print(f"  Breakdown  :")
    for i, cls in enumerate(classes):
        p = probs[i] * 100
        b = '█' * int(p / 5)
        print(f"    {cls:>12} : {b:<20} {p:.1f}%")
    print(f"  Time       : {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"{'─'*52}")

# ── Main ──────────────────────────────────────────────────────────────────────
print(f"[INFO] Opening {PORT} @ {BAUDRATE}...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Connected! Filling window ({WINDOW_SIZE} packets)...\n")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

buffer        = collections.deque(maxlen=WINDOW_SIZE)
packet_count  = 0
last_label    = None
confirm_count = 0

try:
    while True:
        if ser.in_waiting == 0:
            continue

        line = ser.readline().decode('utf-8', errors='replace').rstrip()
        if 'CSI_DATA' not in line:
            continue

        amp = parse_amplitudes(line)
        if amp is None:
            continue

        buffer.append(amp)
        packet_count += 1

        if packet_count <= WINDOW_SIZE and packet_count % 20 == 0:
            print(f"  Buffering: {packet_count}/{WINDOW_SIZE}")

        if len(buffer) < WINDOW_SIZE:
            continue

        # Predict every 10 new packets
        if packet_count % 10 != 0:
            continue

        label, conf, probs = predict(list(buffer))

        if label == last_label:
            confirm_count += 1
        else:
            confirm_count = 1
            last_label    = label

        display(label, conf, probs, confirm_count >= CONFIRM_N)

except KeyboardInterrupt:
    print(f"\n[DONE] Stopped. Packets: {packet_count}")
finally:
    ser.close()