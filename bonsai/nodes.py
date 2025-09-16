from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stat_cols = [c for c in ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'] if c in df.columns]
    if 'Total' not in df.columns and len(stat_cols)==6:
        df['Total'] = df[stat_cols].sum(axis=1)
    drop_cols = [c for c in ['#','Name','Generation'] if c in df.columns]
    return df.drop(columns=drop_cols, errors='ignore')

def split(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    return train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)

def build_pipeline(numeric_cols, cat_cols, rf_params, random_state):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                            remainder='passthrough')
    clf = RandomForestClassifier(random_state=random_state, **rf_params)
    return Pipeline([('prep', pre), ('clf', clf)])

def evaluate(model, X_test, y_true):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }
