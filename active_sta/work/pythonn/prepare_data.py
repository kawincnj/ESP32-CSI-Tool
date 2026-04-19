import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = 'csi_dataset'
OUTPUT_DIR   = 'prepared_data'
WINDOW_SIZE  = 100
STRIDE       = 100
MIN_CSI_LEN  = 64
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Load all CSVs ─────────────────────────────────────────────────────
print("[1/6] Loading CSV files...")
all_dfs = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith('.csv'):
        continue

    fpath = os.path.join(DATASET_DIR, fname)
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"  [SKIP] {fname} — read error: {e}")
        continue

    # Accept either csi_amplitudes or csi_normalized column
    has_amp  = 'csi_amplitudes' in df.columns
    has_norm = 'csi_normalized'  in df.columns

    if 'label' not in df.columns or (not has_amp and not has_norm):
        print(f"  [SKIP] {fname} — missing required columns")
        continue

    # Prefer raw amplitudes; fall back to normalized
    df['csi_source'] = 'csi_amplitudes' if has_amp else 'csi_normalized'

    print(f"  [OK] {fname} — {len(df)} packets | "
          f"label={df['label'].iloc[0]} | "
          f"using={'csi_amplitudes' if has_amp else 'csi_normalized'}")
    all_dfs.append(df)

if not all_dfs:
    raise RuntimeError(f"No valid CSV files found in '{DATASET_DIR}/'")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Total packets : {len(full_df)}")
print(f"  Labels found  : {full_df['label'].unique()}")

# Print per-label count
for label, grp in full_df.groupby('label'):
    print(f"    {label:>15} : {len(grp)} packets")
print()

# ── Step 2: Filter bad packets ────────────────────────────────────────────────
print("[2/6] Filtering bad packets...")
before = len(full_df)

if 'csi_len' in full_df.columns:
    full_df = full_df[full_df['csi_len'] >= MIN_CSI_LEN]

full_df = full_df.reset_index(drop=True)
print(f"  Dropped : {before - len(full_df)} short packets")
print(f"  Remaining: {len(full_df)} packets\n")

# ── Step 3: Parse CSI strings → numpy arrays ──────────────────────────────────
print("[3/6] Parsing CSI arrays...")

def parse_csi(val):
    try:
        arr = np.array(ast.literal_eval(str(val)), dtype=np.float32)
        if len(arr) < MIN_CSI_LEN:
            return None
        return arr[:64]   # keep exactly 64 subcarriers
    except Exception:
        return None

# Use raw amplitudes if available, else normalized
source_col = 'csi_amplitudes' if 'csi_amplitudes' in full_df.columns else 'csi_normalized'
print(f"  Parsing from column: '{source_col}'")

full_df['csi_array'] = full_df[source_col].apply(parse_csi)

before = len(full_df)
full_df = full_df[full_df['csi_array'].notna()].reset_index(drop=True)
print(f"  Dropped unparseable : {before - len(full_df)} rows")
print(f"  Remaining           : {len(full_df)} packets\n")

# ── Step 4: Global normalization (z-score per subcarrier) ─────────────────────
print("[4/6] Global z-score normalization...")

# Stack all into matrix (N, 64)
all_csi     = np.stack(full_df['csi_array'].values)
global_mean = all_csi.mean(axis=0)          # (64,)
global_std  = all_csi.std(axis=0) + 1e-8    # (64,) avoid divide by zero

print(f"  Amplitude range BEFORE norm: {all_csi.min():.3f} to {all_csi.max():.3f}")

# Apply normalization
full_df['csi_array'] = full_df['csi_array'].apply(
    lambda x: ((x - global_mean) / global_std).astype(np.float32)
)

# Verify
normed = np.stack(full_df['csi_array'].values)
print(f"  Amplitude range AFTER  norm: {normed.min():.3f} to {normed.max():.3f}")
print(f"  Mean ≈ {normed.mean():.4f} (should be ~0)")
print(f"  Std  ≈ {normed.std():.4f}  (should be ~1)\n")

# Save normalization stats (needed for inference later)
np.save(os.path.join(OUTPUT_DIR, 'global_mean.npy'), global_mean)
np.save(os.path.join(OUTPUT_DIR, 'global_std.npy'),  global_std)
print(f"  Saved global_mean.npy and global_std.npy\n")

# ── Step 5: Sliding window per label ─────────────────────────────────────────
print("[5/6] Creating sliding windows...")

X_list, y_list = [], []

for label, group in full_df.groupby('label'):
    group      = group.reset_index(drop=True)
    csi_matrix = np.stack(group['csi_array'].values)   # (N, 64)

    window_count = 0
    for start in range(0, len(csi_matrix) - WINDOW_SIZE + 1, STRIDE):
        window = csi_matrix[start : start + WINDOW_SIZE]
        if window.shape == (WINDOW_SIZE, 64):
            X_list.append(window)
            y_list.append(label)
            window_count += 1

    print(f"  [{label:>15}] {len(group):>6} packets → {window_count:>4} windows")

if not X_list:
    raise RuntimeError("No windows created — collect more data per posture!")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print(f"\n  X shape : {X.shape}  → (windows, timesteps, subcarriers)")
print(f"  y shape : {y.shape}\n")

# ── Step 6: Time-based train/val split (NO leakage) ──────────────────────────
print("[6/6] Time-based train/val split...")

# ADD THESE TWO LINES FIRST
encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y).astype(np.int64)

print(f"  Label map : {dict(enumerate(encoder.classes_))}")

X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []

for label_idx, label in enumerate(encoder.classes_):
    mask    = y == label
    X_label = X[mask]
    y_label = y_enc[mask]

    split   = int(len(X_label) * 0.8)

    X_train_list.append(X_label[:split])
    y_train_list.append(y_label[:split])
    X_val_list.append(X_label[split:])
    y_val_list.append(y_label[split:])

    print(f"  [{label:>15}] train: {split} | val: {len(X_label)-split}")

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)
X_val   = np.concatenate(X_val_list)
y_val   = np.concatenate(y_val_list)

# Shuffle train set (but NOT val — keep val in time order)
rng     = np.random.default_rng(42)
shuffle = rng.permutation(len(X_train))
X_train = X_train[shuffle]
y_train = y_train[shuffle]

print(f"\n  Train : {X_train.shape[0]} windows")
print(f"  Val   : {X_val.shape[0]} windows\n")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'),  X_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),    X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'),  y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),    y_val)
np.save(os.path.join(OUTPUT_DIR, 'classes.npy'),  encoder.classes_)

print("─" * 50)
print(f"[DONE] All files saved to '{OUTPUT_DIR}/'")
print(f"  X_train.npy  {X_train.shape}")
print(f"  X_val.npy    {X_val.shape}")
print(f"  y_train.npy  {y_train.shape}")
print(f"  y_val.npy    {y_val.shape}")
print(f"  classes.npy  {encoder.classes_}")
print(f"  global_mean.npy  shape {global_mean.shape}")
print(f"  global_std.npy   shape {global_std.shape}")
print("─" * 50)
print("\nNext step → run train_lstm.py")