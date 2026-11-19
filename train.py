"""Train the DDoS classifier (same steps as DDoS_ML_Classification_Lab.ipynb) and save it.

Usage:
    python train.py --data Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Output:
    models/ddos_model.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TIMING_COLS = ["Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
               "Fwd IAT Min", "Flow Bytes/s", "Flow Packets/s"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    timing = [c for c in TIMING_COLS if c in df.columns]
    df = df[~(df[timing] < 0).any(axis=1)]
    constant = [c for c in df.columns if c != "Label" and df[c].nunique() <= 1]
    df = df.drop(columns=constant)
    if "Fwd Header Length.1" in df.columns and df["Fwd Header Length"].equals(df["Fwd Header Length.1"]):
        df = df.drop(columns=["Fwd Header Length.1"])
    return df


def drop_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = X.corr().abs()
    cols = list(corr.columns)
    to_drop = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > threshold and cols[j] not in to_drop:
                to_drop.append(cols[j])
    return X.drop(columns=to_drop)


def main(data_path: str, out_path: str):
    print("Loading data...")
    df = clean(pd.read_csv(data_path))
    df["Label"] = df["Label"].map({"BENIGN": 0, "DDoS": 1})
    X, y = df.drop(columns="Label"), df["Label"]
    X = drop_correlated(X)
    print(f"Rows: {len(X):,}   Features after cleaning: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    # RFE picks the 20 best features (on scaled data)
    full_scaler = StandardScaler().fit(X_train)
    X_train_sc = pd.DataFrame(full_scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    rfe = RFE(LogisticRegression(max_iter=1000, random_state=SEED), n_features_to_select=20, step=3)
    rfe.fit(X_train_sc, y_train)
    features = X_train_sc.columns[rfe.support_].tolist()
    print("Selected features:", features)

    # Scaler on the selected features only (same numbers as the notebook, just fewer columns)
    scaler = StandardScaler().fit(X_train[features])
    X_train_sel = pd.DataFrame(scaler.transform(X_train[features]), columns=features)
    X_test_sel = pd.DataFrame(scaler.transform(X_test[features]), columns=features)

    # Best settings found by the notebook's grid search
    tuned = {
        "Logistic Regression": LogisticRegression(C=10, max_iter=1000, random_state=SEED),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=20, random_state=SEED, n_jobs=-1),
    }
    model = VotingClassifier([(n, clone(m)) for n, m in tuned.items()], voting="soft")
    print("Training voting ensemble...")
    model.fit(X_train_sel, y_train)

    pred = model.predict(X_test_sel)
    prob = model.predict_proba(X_test_sel)[:, 1]
    metrics = {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
    }
    print("Test metrics:", metrics)

    # One example of each class for the UI buttons (same picks as the LIME cells)
    X_test_raw = X_test[features].reset_index(drop=True)
    y_arr = y_test.to_numpy()
    attack_i = int(np.where((y_arr == 1) & (pred == 1))[0][0])
    benign_ok = np.where((y_arr == 0) & (pred == 0))[0]
    if "Destination Port" in features:  # prefer a typical (non port 80) benign flow as the example
        typical = [k for k in benign_ok if X_test_raw["Destination Port"].iloc[k] != 80]
    else:
        typical = []
    benign_i = int((typical or list(benign_ok))[0])
    examples = {
        "DDoS": {f: float(X_test_raw[f].iloc[attack_i]) for f in features},
        "BENIGN": {f: float(X_test_raw[f].iloc[benign_i]) for f in features},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": features,
                 "examples": examples, "metrics": metrics}, out_path, compress=3)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the CICIDS2017 Friday DDoS CSV")
    ap.add_argument("--out", default="models/ddos_model.joblib")
    args = ap.parse_args()
    main(args.data, args.out)
