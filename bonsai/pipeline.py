import os, json, yaml, pandas as pd, mlflow, mlflow.sklearn, joblib
from bonsai.nodes import clean, split, build_pipeline, evaluate

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

    # target / features
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
    pipe = build_pipeline(prm["numeric_cols"], prm["cat_cols"], prm["rf_params"], prm["random_state"])

    # --- MLflow ---
    mlflow.set_tracking_uri(cat["mlflow_uri"])
    mlflow.set_experiment("pokemon_legendary")

    with mlflow.start_run() as run:
        mlflow.log_params({**prm["rf_params"], "random_state": prm["random_state"]})
        mlflow.sklearn.autolog(log_models=False)
        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log model with signature/example
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_test.iloc[:1])

        run_id = run.info.run_id
        print("MLflow run:", run_id)
        print("Metrics:", metrics)

    # also save a pickle for convenience
    joblib.dump(pipe, cat["model_path"])
    with open(cat["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model →", cat["model_path"])
    print("Saved metrics →", cat["metrics_path"])
    print("MLflow store →", cat["mlflow_uri"])

if __name__ == "__main__":
    main()
