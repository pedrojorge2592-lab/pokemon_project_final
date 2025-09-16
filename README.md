# Pokémon Project – MLOps & Project Management 🐉

A future-proof MLOps mini-project that predicts whether a Pokémon is **Legendary** based on its stats and types.

---

## 🧰 Tech Stack
- Python 3.13
- scikit-learn for ML
- MLflow for experiment tracking
- Streamlit (UI)
- FastAPI (API)
- Docker for packaging

---

## 🚀 Quickstart

### 1) Setup environment
```bash
python -m venv .venv
# Activate
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 2) Add data
data/Pokemon.csv

### 3) Train model
python -m bonsai.pipeline

### 4) Track with MLFlow
mlflow ui --backend-store-uri mlruns


### 5) Run StreamLit App
# Windows
set MODEL_URI=mlruns\<exp_id>\<run_id>\artifacts\model

# macOS/Linux
export MODEL_URI=mlruns/<exp_id>/<run_id>/artifacts/model

streamlit run app.py

### 6) Run FastApi service
uvicorn main:app --reload --port 8000

### 7) Docker
docker build -t pokemon_project:latest .
docker run --rm -p 8501:8501 pokemon_project:latest

