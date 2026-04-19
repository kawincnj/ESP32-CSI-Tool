import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = 'prepared_data'
MODEL_DIR    = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# LSTM hyperparameters
INPUT_SIZE   = 64      # subcarriers per timestep
HIDDEN_SIZE  = 128     # LSTM hidden units
NUM_LAYERS   = 2       # stacked LSTM layers
DROPOUT      = 0.3     # dropout between LSTM layers
BIDIRECTIONAL = True   # BiLSTM — better for CSI patterns

# Training hyperparameters
BATCH_SIZE   = 64
EPOCHS       = 120
LR           = 1e-3
PATIENCE     = 8       # early stopping patience

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {device}")
if device.type == 'cuda':
    print(f"         GPU  : {torch.cuda.get_device_name(0)}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[1/4] Loading prepared data...")
X_train  = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_val    = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_train  = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_val    = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
classes  = np.load(os.path.join(DATA_DIR, 'classes.npy'))

NUM_CLASSES = len(classes)
print(f"  Classes    : {list(classes)}")
print(f"  Train      : {X_train.shape}")
print(f"  Val        : {X_val.shape}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CSIDataset(Dataset):
    def __init__(self, X, y):
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
class PostureLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = INPUT_SIZE,
            hidden_size  = HIDDEN_SIZE,
            num_layers   = NUM_LAYERS,
            batch_first  = True,
            dropout      = DROPOUT,
            bidirectional= BIDIRECTIONAL
        )

        lstm_out_size = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        # x: (batch, timesteps, features)
        out, _ = self.lstm(x)
        out     = out[:, -1, :]      # take last timestep
        return self.classifier(out)

model = PostureLSTM().to(device)
print("[2/4] Model architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Trainable params: {total_params:,}\n")

# ── Training setup ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ── Training loop ─────────────────────────────────────────────────────────────
print("[3/4] Training...\n")

train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []
best_val_loss  = float('inf')
patience_count = 0
best_model_path = os.path.join(MODEL_DIR, 'best_posture_lstm.pth')

for epoch in range(1, EPOCHS + 1):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        t_loss    += loss.item() * len(y_batch)
        t_correct += (logits.argmax(1) == y_batch).sum().item()
        t_total   += len(y_batch)

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)

            v_loss    += loss.item() * len(y_batch)
            v_correct += (logits.argmax(1) == y_batch).sum().item()
            v_total   += len(y_batch)

    # ── Stats ─────────────────────────────────────────────────────────────────
    t_loss_avg = t_loss / t_total
    v_loss_avg = v_loss / v_total
    t_acc      = t_correct / t_total * 100
    v_acc      = v_correct / v_total * 100

    train_losses.append(t_loss_avg)
    val_losses.append(v_loss_avg)
    train_accs.append(t_acc)
    val_accs.append(v_acc)

    scheduler.step(v_loss_avg)

    print(f"Epoch [{epoch:>3}/{EPOCHS}]  "
          f"Train Loss: {t_loss_avg:.4f}  Acc: {t_acc:.1f}%  |  "
          f"Val Loss: {v_loss_avg:.4f}  Acc: {v_acc:.1f}%")

    # ── Early stopping + save best ────────────────────────────────────────────
    if v_loss_avg < best_val_loss:
        best_val_loss = v_loss_avg
        patience_count = 0
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_loss'   : best_val_loss,
            'val_acc'    : v_acc,
            'classes'    : classes,
        }, best_model_path)
        print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs.")
            break

# ── Final evaluation ──────────────────────────────────────────────────────────
print("\n[4/4] Final Evaluation on Validation Set...")

checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        logits = model(X_batch.to(device))
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses,   label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Accuracy curve
axes[1].plot(train_accs, label='Train')
axes[1].plot(val_accs,   label='Val')
axes[1].set_title('Accuracy (%)')
axes[1].set_xlabel('Epoch')
axes[1].legend()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=axes[2])
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=150)
plt.show()

print(f"\n[DONE] Best val accuracy : {checkpoint['val_acc']:.1f}%")
print(f"[DONE] Model saved to    : {best_model_path}")
print(f"[DONE] Plot saved to     : {os.path.join(MODEL_DIR, 'training_results.png')}")