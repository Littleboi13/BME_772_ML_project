# Install required packages in your shell/terminal (not inside a .py script):
# pip install scikit-learn numpy

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import glob
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification # For a sample dataset



# Load all CSV files from Dataset/DATA directory one at a time

data_dir = "Dataset/DATA"
csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))  
if not csv_paths:
    raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

for csv_file in csv_paths:
    # Load a single CSV file
    df = pd.read_csv(csv_file)  # Load the current CSV file

    # Each column is a feature 
    # Load all columns into X (no separate target column)
    X = df.values

    # Print out the csv file name
    print(f"Loaded data from: {csv_file}")

    case = 'seizure' in csv_file
    print(f"Case 'seizure' in filename: {case}")

    # accumulate each CSV as a 2D array and keep filenames/headers; on last file, build a padded 3D array
    if 'data_arrays' not in globals():
        data_arrays = []
        file_names = []
        headers = []

    arr = df.values.astype(object)     # preserve original values (object dtype allows mixed types)
    data_arrays.append(arr)
    file_names.append(os.path.basename(csv_file))
    headers.append(list(df.columns))

    # if this is the last CSV, construct a padded 3D array (files, max_rows, max_cols)
    if csv_file == csv_paths[-1]:
        n_files = len(data_arrays)
        max_rows = max(a.shape[0] for a in data_arrays)
        max_cols = max(a.shape[1] for a in data_arrays)

        # create object array padded with None for missing entries
        data_3d = np.full((n_files, max_rows, max_cols), None, dtype=object)
        for i, a in enumerate(data_arrays):
            r, c = a.shape
            data_3d[i, :r, :c] = a

        # You now have:
        # - data_3d: numpy array shaped (n_files, max_rows, max_cols) with file data (padded with None)
        # - file_names: list of basenames for each file matching axis 0 of data_3d
        # - headers: list of column name lists for each file
        print(f"Created 3D array with shape {data_3d.shape} for files: {file_names}")

# Combine all CSVs into one DataFrame (re-read to keep this block self-contained)
dfs = [pd.read_csv(p) for p in csv_paths]
df_all = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# Determine target column (common names or fall back to last column)
if "target" in df_all.columns:
    y = df_all["target"].values
    X_df = df_all.drop(columns=["target"])
elif "label" in df_all.columns:
    y = df_all["label"].values
    X_df = df_all.drop(columns=["label"])
elif "class" in df_all.columns:
    y = df_all["class"].values
    X_df = df_all.drop(columns=["class"])
else:
    y = df_all.iloc[:, -1].values
    X_df = df_all.iloc[:, :-1]

# Convert categorical features to numeric (one-hot) and get numpy arrays
X = pd.get_dummies(X_df).values

# Encode target if it's non-numeric
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Basic sanity check
n_samples = X.shape[0]
if n_samples < 3:
    raise ValueError("Need at least 3 samples for 3-fold cross-validation")

# 3-fold cross-validation with an SVM classifier
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracy_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = SVC(kernel="rbf", C=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {fold_idx} accuracy: {acc:.4f}")
    accuracy_scores.append(acc)

print(f"Accuracy scores for each fold: {accuracy_scores}")
print(f"Average accuracy across 3 folds: {np.mean(accuracy_scores):.4f}")


# # 1. Load CSV files from the DATA directory (concatenate if multiple)
# data_dir = "DATA"
# csv_paths = glob.glob(os.path.join(data_dir, "*.csv"))
# if not csv_paths:
#     raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

# dfs = [pd.read_csv(p) for p in csv_paths]
# df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# # 1a. Determine target column: use 'target' if present, otherwise assume last column is target
# if "target" in df.columns:
#     y = df["target"].values
#     X_df = df.drop(columns=["target"])
# else:
#     y = df.iloc[:, -1].values
#     X_df = df.iloc[:, :-1]

# # 1b. Convert categorical features to numeric if any, then produce numpy arrays
# X = pd.get_dummies(X_df).values

# # 2. Initialize KFold with 3 folds
# kf = KFold(n_splits=3, shuffle=True, random_state=42)

# # 3. Store accuracy scores for each fold
# accuracy_scores = []

# # 4. Iterate through each fold
# for train_index, test_index in kf.split(X):
#     # Split data into training and test sets for the current fold
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     # Initialize and train your model (e.g., Logistic Regression)
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)

#     # Make predictions on the test set
#     y_pred = model.predict(X_test)

#     # Calculate and store the accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_scores.append(accuracy)

# # 5. Print the accuracy scores for each fold and the average
# print(f"Accuracy scores for each fold: {accuracy_scores}")
# print(f"Average accuracy across 3 folds: {np.mean(accuracy_scores):.4f}")