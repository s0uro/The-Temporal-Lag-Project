[![Netlify Status](https://api.netlify.com/api/v1/badges/890882d4-bcc5-48f2-81ce-d88862806ef5/deploy-status)](https://app.netlify.com/projects/the-temporal-lag-project/deploys)

## The Temporal Lag Project

Interactive temporal lag analysis with ML models and SHAP explainability.

### Local run (Flask backend + static frontend)

- **Backend**: `python backend/api.py` (serves on `http://127.0.0.1:5001`)
- **Frontend**: open `http://127.0.0.1:5001/` in your browser.

### Streamlit Cloud deployment

- Main file: `streamlit_app.py`
- Requirements: `requirements.txt`

### Frontend-only deployment (Netlify / Vercel)

- Deploy the `frontend/` folder as a static site.
- Point API calls to the deployed backend URL (e.g. Render/Railway).

