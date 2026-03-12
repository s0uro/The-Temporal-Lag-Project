import pandas as pd
from pathlib import Path
from joblib import load

import matplotlib

# Χρησιμοποιούμε non-GUI backend ώστε να δουλεύει σε server/threads
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split


INPUT_FILE = "backend/data_clean.csv"
MODEL_XGB_PATH = "backend/model_xgb.joblib"
SPLIT_INFO_PATH = "backend/split_info.joblib"

TARGET_COL = "target"


def read_input(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Μη υποστηριζόμενο format αρχείου: {ext}")


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Λείπει η στήλη target ('{TARGET_COL}') από τα δεδομένα.")

    df = df.copy()

    # Ίδια προεπεξεργασία features όπως στο training.py / evaluation.py
    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype == "object":
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_na_ratio = numeric.notna().mean()
            if non_na_ratio > 0.5:
                df[col] = numeric
            else:
                df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=[TARGET_COL])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    X = X.fillna(0)

    y = df[TARGET_COL]
    return X, y


def main() -> None:
    print("Φόρτωση δεδομένων και XGBoost...")
    df = read_input(INPUT_FILE)
    X, y = get_feature_target(df)

    xgb_model = load(MODEL_XGB_PATH)
    split_info = load(SPLIT_INFO_PATH)

    test_size = split_info.get("test_size", 0.2)
    random_state = split_info.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Για SHAP, χρησιμοποιούμε ένα υποσύνολο για να είναι πιο γρήγορο
    background = X_train.sample(min(200, len(X_train)), random_state=42)

    print("Δημιουργία SHAP TreeExplainer για XGBoost...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(background)

    # === 1. Συνολική σημασία χαρακτηριστικών (summary plot) ===
    print("Αποθήκευση συνολικού summary plot σε 'shap_summary.png'...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        background,
        show=False,
        plot_type="dot",
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200)
    plt.close()

    # === 2. Ειδικά για τη μεταβλητή Lag (Lag vs probability) ===
    if "Lag" in background.columns:
        print("Αποθήκευση dependence plot για 'Lag' σε 'shap_dependence_Lag.png'...")
        plt.figure()
        shap.dependence_plot(
            "Lag",
            shap_values,
            background,
            show=False,
        )
        plt.tight_layout()
        plt.savefig("shap_dependence_Lag.png", dpi=200)
        plt.close()
    else:
        print("Προειδοποίηση: Δεν βρέθηκε η στήλη 'Lag' στα features για SHAP dependence plot.")

    print("Ολοκληρώθηκαν τα SHAP γραφήματα (summary + Lag dependence).")


if __name__ == "__main__":
    main()

