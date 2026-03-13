from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory  # pyright: ignore[reportMissingImports]

from backend import (
    preprocessing,
    cross_correlation,
    training,
    evaluation,
    shap_explainability,
)


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "users.db"

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="",
)


def init_db() -> None:
    """Δημιουργεί τον πίνακα users αν δεν υπάρχει."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()


@app.route("/")
def index() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/metrics.json")
def serve_metrics() -> object:
    return send_from_directory(str(BASE_DIR), "metrics.json")


@app.route("/lag_stats.json")
def serve_lag_stats() -> object:
    return send_from_directory(str(BASE_DIR), "lag_stats.json")


@app.route("/shap_summary.png")
def serve_shap_summary() -> object:
    return send_from_directory(str(BASE_DIR), "shap_summary.png")


@app.route("/shap_dependence_Lag.png")
def serve_shap_lag() -> object:
    return send_from_directory(str(BASE_DIR), "shap_dependence_Lag.png")


@app.post("/api/auth/signup")
def signup() -> object:
    """Απλό endpoint για αποθήκευση νέου χρήστη σε SQLite."""
    data = request.get_json(silent=True) or {}

    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not (first_name and last_name and email and password):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "All fields are required.",
                }
            ),
            400,
        )

    # Πολύ απλό hash κωδικού (για demo). Σε πραγματική εφαρμογή θα χρησιμοποιούσαμε bcrypt/argon2.
    password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    created_at = datetime.utcnow().isoformat(timespec="seconds")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users (first_name, last_name, email, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (first_name, last_name, email, password_hash, created_at),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "This email is already registered.",
                }
            ),
            409,
        )

    return jsonify({"status": "ok", "message": "User registered successfully."})


@app.get("/api/auth/users")
def list_users() -> object:
    """Επιστρέφει λίστα χρηστών (id, όνομα, email, created_at)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, first_name, last_name, email, created_at FROM users ORDER BY created_at DESC LIMIT 100"
        ).fetchall()

    users = [
        {
            "id": row["id"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "email": row["email"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return jsonify({"status": "ok", "users": users})


@app.post("/api/run")
def run_full_pipeline() -> object:
    """
    Δέχεται CSV/XLSX, το αποθηκεύει ως backend/data.(csv|xlsx),
    τρέχει όλο το pipeline (preprocessing → cross_corr → training → evaluation → SHAP)
    και επιστρέφει τα metrics/lag_stats ως JSON.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Unsupported file type. Use .csv or .xlsx",
                }
            ),
            400,
        )

    # Αποθήκευση input δεδομένων μέσα στο backend
    target_name = "data.xlsx" if ext in {".xlsx", ".xls"} else "data.csv"
    target_path = BACKEND_DIR / target_name
    file.save(target_path)

    try:
        # Τρέξε όλο το pipeline με βάση τα υπάρχοντα modules
        preprocessing.main()
        cross_correlation.main()
        training.main()
        evaluation.main()
        shap_explainability.main()
    except Exception as exc:  # pragma: no cover - απλό error surface
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Pipeline failed: {exc!s}",
                }
            ),
            500,
        )

    # Διάβασε τα JSON/αρχεία που παρήχθησαν
    metrics_path = BASE_DIR / "metrics.json"
    lag_path = BASE_DIR / "lag_stats.json"

    result: dict[str, object] = {"status": "ok"}

    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if lag_path.exists():
        result["lag_stats"] = json.loads(lag_path.read_text(encoding="utf-8"))

    return jsonify(result)


if __name__ == "__main__":
    # Τρέχει σε http://127.0.0.1:5001/
    # Χωρίς debug/reloader για να μην μπλέκει με matplotlib/threads
    app.run(debug=False, port=5001)

