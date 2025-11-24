import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# =========================
# 1. Load all CSV files, derive labels from filenames
# =========================

data_dir = "Dataset/DATA"
csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))

if not csv_paths:
    raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

dfs = []
data_arrays = []
file_names = []
headers = []

for csv_file in csv_paths:
    # Load current CSV file
    df = pd.read_csv(csv_file)

    # Determine label from filename: 1 = seizure, 0 = non-seizure
    is_seizure = "seizure" in os.path.basename(csv_file).lower()
    label_value = 1 if is_seizure else 0
    df["label"] = label_value

    print(f"Loaded data from: {csv_file}")
    print(f"  → Case 'seizure' in filename: {is_seizure}")

    dfs.append(df)

    # Optional: keep raw arrays for 3D array construction (without label)
    arr = df.drop(columns=["label"]).values.astype(object)
    data_arrays.append(arr)
    file_names.append(os.path.basename(csv_file))
    headers.append(list(df.drop(columns=["label"]).columns))

# =========================
# 2. Optional: build padded 3D array of raw data (without labels)
# =========================

n_files = len(data_arrays)
max_rows = max(a.shape[0] for a in data_arrays)
max_cols = max(a.shape[1] for a in data_arrays)

data_3d = np.full((n_files, max_rows, max_cols), None, dtype=object)
for i, a in enumerate(data_arrays):
    r, c = a.shape
    data_3d[i, :r, :c] = a

print(f"Created 3D array with shape {data_3d.shape} for files: {file_names}")

# =========================
# 3. Build a single dataset (X, y) for ML
# =========================

# Concatenate all CSVs (with labels) into one DataFrame
df_all = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# Separate features and label
y = df_all["label"].values.astype(int)
X_df = df_all.drop(columns=["label"])

print("Original feature columns:", list(X_df.columns))
print("Example of label values (0 = non-seizure, 1 = seizure):")
print(pd.Series(y).value_counts())

# =========================
# 4. Expand semicolon-encoded feature columns and convert to numeric
# =========================

X_df_clean = X_df.copy()

for col in list(X_df_clean.columns):
    if X_df_clean[col].dtype == object:
        # Check if this column has semicolon-separated values
        has_semicolon = X_df_clean[col].astype(str).str.contains(";").any()

        if has_semicolon:
            # Split into multiple columns
            expanded = X_df_clean[col].astype(str).str.split(";", expand=True)
            # Convert each split part to numeric
            expanded = expanded.apply(pd.to_numeric, errors="coerce")
            # Give them unique names
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]

            # Drop original string column and add expanded numeric columns
            X_df_clean = pd.concat(
                [X_df_clean.drop(columns=[col]), expanded],
                axis=1
            )
            print(f"Expanded column '{col}' into {expanded.shape[1]} numeric columns.")
        else:
            # Try to convert plain object column to numeric
            X_df_clean[col] = pd.to_numeric(X_df_clean[col], errors="coerce")

print("Final feature columns after cleaning:", len(X_df_clean.columns))

X = X_df_clean.values  # still may contain NaNs; imputer will handle them

# Sanity check: at least 2 classes
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(
        f"Need at least 2 classes for classification, but found only: {unique_classes}"
    )

n_samples, n_features = X.shape
print(f"Total samples: {n_samples}, features: {n_features}")
print("Class distribution:", {cls: (y == cls).sum() for cls in unique_classes})

if n_samples < 3:
    raise ValueError("Need at least 3 samples for 3-fold cross-validation")

# Quick diagnostic: NaNs in features
nan_count = np.isnan(X).sum()
print("NaNs in X before imputation:", nan_count)

# =========================
# 5. Stratified 3-fold CV with Logistic Regression + Imputer
# =========================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # handle NaNs in features
    ("clf", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1
    ))
])

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

accuracy_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold_idx} accuracy: {acc:.4f}")
    accuracy_scores.append(acc)

print("Accuracy scores for each fold:", accuracy_scores)
print(f"Average accuracy across 3 folds: {np.mean(accuracy_scores):.4f}")
