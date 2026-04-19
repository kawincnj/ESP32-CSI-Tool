import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = 'prepared_data'
MODEL_DIR   = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Model hyperparameters
INPUT_SIZE   = 64      # subcarriers
CNN_CHANNELS = [32, 64, 128]   # conv layer output channels
KERNEL_SIZE  = 3
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
ATTN_HEADS   = 4       # must divide LSTM_HIDDEN*2 evenly
DROPOUT      = 0.3

# Training
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
PATIENCE     = 10

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] {device}")
if device.type == 'cuda':
    print(f"         {torch.cuda.get_device_name(0)}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[1/4] Loading data...")
X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
classes = np.load(os.path.join(DATA_DIR, 'classes.npy'))
NUM_CLASSES = len(classes)

print(f"  Classes : {list(classes)}")
print(f"  Train   : {X_train.shape}")
print(f"  Val     : {X_val.shape}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CSIDataset(Dataset):
    def __init__(self, X, y):
        # CNN expects (batch, channels, timesteps, features)
        # We treat subcarriers as features, add channel dim
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CSIDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(CSIDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=(device.type == 'cuda'))

# ── Model ─────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style attention over CNN channels."""
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
        # x: (batch, channels, time)
        w = self.pool(x).squeeze(-1)   # (batch, channels)
        w = self.fc(w).unsqueeze(-1)   # (batch, channels, 1)
        return x * w                   # scale channels


class TemporalAttention(nn.Module):
    """Multi-head self-attention over LSTM timesteps."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_size,
            num_heads   = num_heads,
            dropout     = 0.1,
            batch_first = True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, timesteps, hidden)
        attn_out, attn_weights = self.attn(x, x, x)
        out = self.norm(x + attn_out)   # residual connection
        return out, attn_weights


class CNNBlock(nn.Module):
    """1D CNN block: Conv → BN → ReLU → ChannelAttention → Dropout"""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.ca      = ChannelAttention(out_ch)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)
        return self.dropout(x)


class CNNLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # ── CNN feature extractor ──────────────────────────────────────────
        # Input: (batch, timesteps, 64)
        # Permute to (batch, 64, timesteps) for Conv1d
        cnn_layers = []
        in_ch = INPUT_SIZE
        for out_ch in CNN_CHANNELS:
            cnn_layers.append(CNNBlock(in_ch, out_ch, KERNEL_SIZE))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        # Output: (batch, 128, timesteps)

        # ── BiLSTM ────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = CNN_CHANNELS[-1],   # 128
            hidden_size  = LSTM_HIDDEN,         # 128
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
            dropout      = DROPOUT,
            bidirectional= True
        )
        lstm_out = LSTM_HIDDEN * 2   # 256 (bidirectional)

        # ── Temporal Attention ────────────────────────────────────────────
        self.attn = TemporalAttention(lstm_out, ATTN_HEADS)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        # x: (batch, timesteps=100, features=64)

        # CNN: needs (batch, features, timesteps)
        x = x.permute(0, 2, 1)          # → (batch, 64, 100)
        x = self.cnn(x)                  # → (batch, 128, 100)
        x = x.permute(0, 2, 1)          # → (batch, 100, 128)

        # BiLSTM
        x, _ = self.lstm(x)              # → (batch, 100, 256)

        # Temporal Attention
        x, attn_weights = self.attn(x)  # → (batch, 100, 256)

        # Global average pool over timesteps
        x = x.mean(dim=1)               # → (batch, 256)

        return self.classifier(x)        # → (batch, num_classes)


# ── Init model ────────────────────────────────────────────────────────────────
model = CNNLSTMAttention().to(device)
print("[2/4] Model:")
print(model)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Trainable params: {total:,}\n")

# ── Training setup ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ── Training loop ─────────────────────────────────────────────────────────────
print("[3/4] Training...\n")

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss  = float('inf')
patience_count = 0
best_path      = os.path.join(MODEL_DIR, 'best_cnn_lstm_attn.pth')

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    t_loss = t_correct = t_total = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss    += loss.item() * len(y_b)
        t_correct += (logits.argmax(1) == y_b).sum().item()
        t_total   += len(y_b)

    # Validate
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            logits    = model(X_b)
            loss      = criterion(logits, y_b)
            v_loss    += loss.item() * len(y_b)
            v_correct += (logits.argmax(1) == y_b).sum().item()
            v_total   += len(y_b)

    t_l = t_loss / t_total
    v_l = v_loss / v_total
    t_a = t_correct / t_total * 100
    v_a = v_correct / v_total * 100

    train_losses.append(t_l)
    val_losses.append(v_l)
    train_accs.append(t_a)
    val_accs.append(v_a)

    scheduler.step()
    lr_now = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch:>3}/{EPOCHS}]  "
          f"Train: {t_l:.4f} / {t_a:.1f}%  |  "
          f"Val: {v_l:.4f} / {v_a:.1f}%  |  "
          f"LR: {lr_now:.6f}")

    if v_l < best_val_loss:
        best_val_loss  = v_l
        patience_count = 0
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_loss'   : best_val_loss,
            'val_acc'    : v_a,
            'classes'    : classes,
            'config': {
                'input_size'   : INPUT_SIZE,
                'cnn_channels' : CNN_CHANNELS,
                'kernel_size'  : KERNEL_SIZE,
                'lstm_hidden'  : LSTM_HIDDEN,
                'lstm_layers'  : LSTM_LAYERS,
                'attn_heads'   : ATTN_HEADS,
                'dropout'      : DROPOUT,
                'num_classes'  : NUM_CLASSES,
            }
        }, best_path)
        print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs.")
            break

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[4/4] Final Evaluation...")
ckpt = torch.load(best_path, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

preds, labels = [], []
with torch.no_grad():
    for X_b, y_b in val_loader:
        out = model(X_b.to(device)).argmax(1).cpu().numpy()
        preds.extend(out)
        labels.extend(y_b.numpy())

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=classes))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CNN-LSTM + Attention Training Results', fontsize=14)

axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses,   label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(train_accs, label='Train')
axes[1].plot(val_accs,   label='Val')
axes[1].set_title('Accuracy (%)')
axes[1].set_xlabel('Epoch')
axes[1].legend()

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=axes[2])
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
out_img = os.path.join(MODEL_DIR, 'cnn_lstm_results.png')
plt.savefig(out_img, dpi=150)
plt.show()

print(f"\n[DONE] Best val accuracy : {ckpt['val_acc']:.1f}%")
print(f"[DONE] Model saved       : {best_path}")
print(f"[DONE] Plot saved        : {out_img}")
print("\nNext → run predict_realtime.py")