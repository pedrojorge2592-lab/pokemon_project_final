import os
import mlflow
import streamlit as st
import pandas as pd
from mlflow.tracking import MlflowClient

# Optional: for single-file artifacts (.pkl)
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Pokémon Legendary Predictor")

# --- Config from env ---
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
EXPERIMENT_NAME = os.getenv("MODEL_EXPERIMENT", "pokemon_legendary")
MODEL_URI = (os.getenv("MODEL_URI", "") or "").strip()

# Make sure MLflow resolves to the same store the pipeline writes to
mlflow.set_tracking_uri(TRACKING_URI)

def _load_any_model(uri: str):
    """Try MLflow pyfunc first; if that fails, treat as file artifact and load with joblib."""
    # Absolute path or file://? -> direct joblib
    if uri.startswith("/") or uri.startswith("file://"):
        if joblib is None:
            raise RuntimeError("joblib missing; cannot load file artifact")
        p = uri.replace("file://", "", 1)
        return joblib.load(p)

    # Try MLflow model dir
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception:
        # Try to download artifact and load with joblib
        local = mlflow.artifacts.download_artifacts(uri)
        if joblib is None:
            raise RuntimeError("joblib missing; cannot load file artifact")
        return joblib.load(local)

def _latest_run_ids(exp_name: str):
    """Return a list of newest-to-oldest finished run IDs for the experiment."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=25,
    )
    return [r.info.run_id for r in runs]

def _auto_pick_model_uri():
    """Find newest usable artifact among recent runs."""
    run_ids = _latest_run_ids(EXPERIMENT_NAME)
    for rid in run_ids:
        # Prefer the proper MLflow model directory (what your pipeline logs)
        candidates = (
            f"runs:/{rid}/model",                  # mlflow.sklearn.log_model(..., artifact_path='model')
            f"runs:/{rid}/rf_model.pkl",           # if someone logged a top-level pickle
            f"runs:/{rid}/artifacts/rf_model.pkl"  # if someone logged it under 'artifacts/'
        )
        for candidate in candidates:
            try:
                _ = _load_any_model(candidate)  # probe
                return candidate, rid
            except Exception:
                continue
    return None, None

@st.cache_resource(show_spinner=True)
def get_model_and_meta(exp_name: str, model_uri: str | None):
    """Cache the loaded model + metadata across reruns."""
    chosen_uri = (model_uri or "").strip()
    chosen_run = None

    if not chosen_uri or chosen_uri.lower() in {"auto", "latest"}:
        chosen_uri, chosen_run = _auto_pick_model_uri()
        if not chosen_uri:
            raise RuntimeError(
                "Couldn't find a usable model. Train a run in MLflow that logs "
                "either 'model/' (mlflow.sklearn.log_model) or 'rf_model.pkl' as an artifact."
            )
    # If user pasted a specific URI, we still want to know the run id (best effort)
    if chosen_uri and chosen_run is None and chosen_uri.startswith("runs:/"):
        try:
            chosen_run = chosen_uri.split("/")[2]
        except Exception:
            chosen_run = None

    model = _load_any_model(chosen_uri)
    return model, chosen_uri, chosen_run

# UI sidebar: show config + refresh
with st.sidebar:
    st.caption("MLflow")
    st.code(f"Tracking: {TRACKING_URI}")
    st.code(f"Experiment: {EXPERIMENT_NAME}")
    st.code(f"MODEL_URI: {MODEL_URI or 'auto (latest)'}")
    if st.button("🔄 Refresh model (pick newest)"):
        # Clear the cached model so next call re-picks
        get_model_and_meta.clear()
        # Streamlit 1.32+: st.rerun(); older versions had experimental_rerun()
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

# Try to load a model
model = None
chosen_uri = None
chosen_run = None
try:
    model, chosen_uri, chosen_run = get_model_and_meta(EXPERIMENT_NAME, MODEL_URI)
    st.success(f"Loaded model from: {chosen_uri}")
    if chosen_run:
        st.caption(f"Run ID: {chosen_run}")
except Exception as e:
    st.error(f"Failed to load a model.\n\n{e}")

st.title("Is this Pokémon Legendary?")

col1, col2, col3 = st.columns(3)
HP = col1.number_input("HP", 1, 255, 60)
Attack = col1.number_input("Attack", 1, 255, 70)
Defense = col1.number_input("Defense", 1, 255, 65)
SpA = col2.number_input("Sp. Atk", 1, 255, 70)
SpD = col2.number_input("Sp. Def", 1, 255, 70)
Speed = col2.number_input("Speed", 1, 255, 70)
Total = col3.number_input("Total", 6, 1125, HP + Attack + Defense + SpA + SpD + Speed)
Type1 = col3.text_input("Type 1", "Water")
Type2 = col3.text_input("Type 2 (optional)", "") or None

if st.button("Predict") and model is not None:
    X = pd.DataFrame([{
        "HP": HP, "Attack": Attack, "Defense": Defense,
        "Sp. Atk": SpA, "Sp. Def": SpD, "Speed": Speed,
        "Total": Total, "Type 1": Type1, "Type 2": Type2
    }])

    # Prefer predict_proba, fallback to predict
    prob = None
    try:
        proba = model.predict_proba(X)
        prob = float(proba[0][1]) if getattr(proba, "shape", (0, 0))[1] >= 2 else float(proba[0])
    except Exception:
        y = model.predict(X)
        val = float(y[0]) if hasattr(y, "__len__") else float(y)
        prob = val if 0.0 <= val <= 1.0 else (1.0 if val >= 0.5 else 0.0)

    st.metric("Legendary probability", f"{prob:.2%}")
    st.write("Prediction:", "⭐ Legendary" if prob >= 0.5 else "Not Legendary")
