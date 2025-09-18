# pipeline.py
import os
import json
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from bonsai.nodes import clean, build_pipeline, evaluate

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # --- config ---
    cat = load_yaml("conf/catalog.yml")
    prm = load_yaml("conf/parameters.yml")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs(os.path.dirname(cat["pokemon_clean_path"]), exist_ok=True)

    # --- data ---
    df = pd.read_csv(cat["pokemon_raw"]["filepath"])
    df = clean(df)

    target_col = prm["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    X = df[prm["numeric_cols"] + prm["cat_cols"]]
    y = df[target_col].astype(int)

    # --- split ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=prm["test_size"], stratify=y, random_state=prm["random_state"]
    )

    # save splits (optional)
    pd.concat([X_train, y_train], axis=1).to_parquet(cat["train_path"])
    pd.concat([X_test, y_test], axis=1).to_parquet(cat["test_path"])

    # --- model ---
    pipe = build_pipeline(
        prm["numeric_cols"], prm["cat_cols"], prm["rf_params"], prm["random_state"]
    )

    # ========== MLflow config (IMPORTANT) ==========
    # Use the same tracking URI as the app (compose sets this env)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    # Keep experiment name consistent with the app
    experiment_name = os.getenv("MODEL_EXPERIMENT", "pokemon_legendary")
    mlflow.set_experiment(experiment_name)
    # ===============================================

    with mlflow.start_run() as run:
        mlflow.log_params({**prm["rf_params"], "random_state": prm["random_state"]})

        # Don't autolog models automatically (we will log explicitly)
        mlflow.sklearn.autolog(log_models=False)

        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)

        # 1) Log an MLflow MODEL directory (what the app prefers)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_test.iloc[:1],
        )

        # 2) Also save a convenience pickle and log it as an artifact
        pickle_path = cat["model_path"]  # e.g., "artifacts/rf_model.pkl"
        joblib.dump(pipe, pickle_path)
        mlflow.log_artifact(pickle_path)  # attach the file to this run

        # Save metrics file and log it too (optional)
        metrics_path = cat["metrics_path"]  # e.g., "artifacts/metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print("MLflow run:", run.info.run_id)
        print("Metrics:", metrics)
        print("Tracking URI:", tracking_uri)
        print("Experiment:", experiment_name)

    print("Saved model →", pickle_path)
    print("Saved metrics →", metrics_path)

if __name__ == "__main__":
    main()
