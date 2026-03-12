from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

import backend.preprocessing as preprocessing
import backend.cross_correlation as cross_correlation
import backend.training as training
import backend.evaluation as evaluation
import backend.shap_explainability as shap_explainability


BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"


def run_full_pipeline(uploaded_path: Path) -> dict[str, Any]:
    """
    Αποθηκεύει το ανεβασμένο αρχείο μέσα στο backend/,
    τρέχει preprocessing → cross_correlation → training → evaluation → SHAP
    και επιστρέφει metrics/lag_stats ως dict.
    """
    # Προσδιορισμός ονόματος-στόχου όπως στο Flask API
    ext = uploaded_path.suffix.lower()
    target_name = "data.xlsx" if ext in {".xlsx", ".xls"} else "data.csv"
    target_path = BACKEND_DIR / target_name

    # Αντιγραφή/μετακίνηση αρχείου στο backend/
    if uploaded_path != target_path:
        target_path.write_bytes(uploaded_path.read_bytes())

    # Τρέξε όλο το pipeline με βάση τα υπάρχοντα modules
    preprocessing.main()
    cross_correlation.main()
    training.main()
    evaluation.main()
    shap_explainability.main()

    # Διάβασε τα JSON/αρχεία που παρήχθησαν
    metrics_path = BASE_DIR / "metrics.json"
    lag_path = BASE_DIR / "lag_stats.json"

    result: dict[str, Any] = {}
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if lag_path.exists():
        result["lag_stats"] = json.loads(lag_path.read_text(encoding="utf-8"))

    return result


def main() -> None:
    st.set_page_config(
        page_title="Temporal Lag Project",
        layout="wide",
    )

    st.title("The Temporal Lag Project")
    st.markdown(
        "Ανέβασε ένα CSV ή Excel αρχείο για να τρέξει όλο το pipeline "
        "(preprocessing → training → evaluation → SHAP)."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV/XLSX", type=["csv", "xlsx", "xls"]
    )

    run_button = st.button("Run pipeline", disabled=uploaded_file is None)

    if run_button:
        if not uploaded_file:
            st.warning("Πρώτα επίλεξε ένα αρχείο.")
            return

        # Αποθήκευση προσωρινού αρχείου στο backend/
        tmp_path = BACKEND_DIR / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.getbuffer())

        with st.spinner("Τρέχει όλο το pipeline..."):
            try:
                result = run_full_pipeline(tmp_path)
            except Exception as exc:  # pragma: no cover
                st.error(f"Pipeline failed: {exc!s}")
                return

        st.success("Το pipeline ολοκληρώθηκε επιτυχώς!")

        # === Εμφάνιση metrics ===
        metrics = result.get("metrics")
        if isinstance(metrics, dict):
            st.subheader("Model performance")

            col1, col2 = st.columns(2)
            with col1:
                xgb = metrics.get("xgboost", {})
                st.markdown("**XGBoost**")
                st.metric("Accuracy", f"{xgb.get('accuracy', 0):.4f}")
                st.metric("F1", f"{xgb.get('f1', 0):.4f}")

            with col2:
                mlp = metrics.get("mlp", {})
                st.markdown("**MLP**")
                st.metric("Accuracy", f"{mlp.get('accuracy', 0):.4f}")
                st.metric("F1", f"{mlp.get('f1', 0):.4f}")

            # Confusion matrices (αν υπάρχουν)
            st.markdown("### Confusion matrices")
            cm_cols = st.columns(2)
            with cm_cols[0]:
                cm_xgb = metrics.get("xgboost", {}).get("confusion_matrix")
                if cm_xgb:
                    st.markdown("**XGBoost**")
                    st.table(cm_xgb)
            with cm_cols[1]:
                cm_mlp = metrics.get("mlp", {}).get("confusion_matrix")
                if cm_mlp:
                    st.markdown("**MLP**")
                    st.table(cm_mlp)

        # === Εμφάνιση lag stats ===
        lag_stats = result.get("lag_stats")
        if isinstance(lag_stats, dict):
            st.subheader("Lag statistics")
            cols = st.columns(3)
            cols[0].metric("Mean", f"{lag_stats.get('mean', 0):.4f}")
            cols[1].metric("Median", f"{lag_stats.get('median', 0):.4f}")
            cols[2].metric("Std", f"{lag_stats.get('std', 0):.4f}")

        # === SHAP εικόνες ===
        shap_summary_path = BASE_DIR / "shap_summary.png"
        shap_lag_path = BASE_DIR / "shap_dependence_Lag.png"

        st.subheader("SHAP explainability")
        shap_cols = st.columns(2)
        with shap_cols[0]:
            if shap_summary_path.exists():
                st.markdown("**Summary plot**")
                st.image(str(shap_summary_path))
        with shap_cols[1]:
            if shap_lag_path.exists():
                st.markdown("**Lag dependence plot**")
                st.image(str(shap_lag_path))


if __name__ == "__main__":
    main()

