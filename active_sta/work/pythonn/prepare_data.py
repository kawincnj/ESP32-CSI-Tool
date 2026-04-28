import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
# Get the project root directory (2 levels up from this script)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATASET_DIR  = os.path.join(PROJECT_ROOT, 'csi_dataset')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'prepared_data')
WINDOW_SIZE  = 100      # timesteps per sequence
STRIDE       = 100      # no overlap = no leakage
MIN_CSI_LEN  = 64
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load CSVs ─────────────────────────────────────────────────────────
print("[1/6] Loading CSV files...")
all_dfs = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith('.csv'):
        continue
    fpath = os.path.join(DATASET_DIR, fname)
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  [SKIP] {fname} — {e}")
        continue

    has_amp  = 'csi_amplitudes' in df.columns
    has_norm = 'csi_normalized'  in df.columns
    if 'label' not in df.columns or (not has_amp and not has_norm):
        print(f"  [SKIP] {fname} — missing columns")
        continue

    print(f"  [OK] {fname} — {len(df)} packets | label={df['label'].iloc[0]}")
    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError(f"No valid CSVs in '{DATASET_DIR}/'")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Total packets : {len(full_df)}")
for label, grp in full_df.groupby('label'):
    print(f"    {label:>15} : {len(grp)} packets")

# ── Step 2: Filter ────────────────────────────────────────────────────────────
print("\n[2/6] Filtering...")
before = len(full_df)
if 'csi_len' in full_df.columns:
    full_df = full_df[full_df['csi_len'] >= MIN_CSI_LEN]
full_df = full_df.reset_index(drop=True)
print(f"  Dropped: {before - len(full_df)} | Remaining: {len(full_df)}")

# ── Step 3: Parse CSI ─────────────────────────────────────────────────────────
print("\n[3/6] Parsing CSI arrays...")

def parse_csi(val):
    try:
        arr = np.array(ast.literal_eval(str(val)), dtype=np.float32)
        return arr[:64] if len(arr) >= 64 else None
    except Exception:
        return None

col = 'csi_amplitudes' if 'csi_amplitudes' in full_df.columns else 'csi_normalized'
print(f"  Using column: '{col}'")
full_df['csi_array'] = full_df[col].apply(parse_csi)
before = len(full_df)
full_df = full_df[full_df['csi_array'].notna()].reset_index(drop=True)
print(f"  Dropped: {before - len(full_df)} | Remaining: {len(full_df)}")

# ── Step 4: Global normalization ──────────────────────────────────────────────
print("\n[4/6] Global z-score normalization...")
all_csi     = np.stack(full_df['csi_array'].values)
global_mean = all_csi.mean(axis=0)
global_std  = all_csi.std(axis=0) + 1e-8

print(f"  Before: {all_csi.min():.3f} to {all_csi.max():.3f}")
full_df['csi_array'] = full_df['csi_array'].apply(
    lambda x: ((x - global_mean) / global_std).astype(np.float32)
)
normed = np.stack(full_df['csi_array'].values)
print(f"  After : {normed.min():.3f} to {normed.max():.3f}")
print(f"  Mean  : {normed.mean():.4f} | Std: {normed.std():.4f}")

np.save(os.path.join(OUTPUT_DIR, 'global_mean.npy'), global_mean)
np.save(os.path.join(OUTPUT_DIR, 'global_std.npy'),  global_std)

# ── Step 5: Sliding windows ───────────────────────────────────────────────────
print("\n[5/6] Creating windows...")
X_list, y_list = [], []

for label, group in full_df.groupby('label'):
    group      = group.reset_index(drop=True)
    csi_matrix = np.stack(group['csi_array'].values)
    count      = 0
    for start in range(0, len(csi_matrix) - WINDOW_SIZE + 1, STRIDE):
        window = csi_matrix[start:start + WINDOW_SIZE]
        if window.shape == (WINDOW_SIZE, 64):
            X_list.append(window)
            y_list.append(label)
            count += 1
    print(f"  [{label:>15}] {len(group)} packets → {count} windows")

if not X_list:
    raise RuntimeError("No windows created — collect more data!")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)
print(f"\n  X: {X.shape} | y: {y.shape}")

# ── Step 6: Time-based split ──────────────────────────────────────────────────
print("\n[6/6] Time-based train/val split...")

encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y).astype(np.int64)
print(f"  Label map: {dict(enumerate(encoder.classes_))}")

X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []

for label in encoder.classes_:
    mask    = y == label
    X_lbl   = X[mask]
    y_lbl   = y_enc[mask]
    split   = int(len(X_lbl) * 0.8)

    X_train_list.append(X_lbl[:split])
    y_train_list.append(y_lbl[:split])
    X_val_list.append(X_lbl[split:])
    y_val_list.append(y_lbl[split:])
    print(f"  [{label:>15}] train: {split} | val: {len(X_lbl)-split}")

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_val   = np.concatenate(X_val_list)
y_val   = np.concatenate(y_val_list)

# Shuffle train only
rng     = np.random.default_rng(RANDOM_SEED)
shuffle = rng.permutation(len(X_train))
X_train = X_train[shuffle]
y_train = y_train[shuffle]

print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'),  X_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),    X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'),  y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),    y_val)
np.save(os.path.join(OUTPUT_DIR, 'classes.npy'),  encoder.classes_)

print(f"\n[DONE] Saved to '{OUTPUT_DIR}/'")
print(f"  X_train: {X_train.shape}")
print(f"  X_val  : {X_val.shape}")
print("\nNext → run train_cnn_lstm.py")