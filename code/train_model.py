from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
MISSING_THRESHOLD = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S1_FILE = DATA_DIR / "Table_S1_dataset.csv"
S4_FILE = DATA_DIR / "Table_S4_descriptor_matrix.csv"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    if not S1_FILE.exists():
        raise FileNotFoundError(f"Missing file: {S1_FILE}")
    if not S4_FILE.exists():
        raise FileNotFoundError(f"Missing file: {S4_FILE}")

    s1 = pd.read_csv(S1_FILE)
    s4 = pd.read_csv(S4_FILE)

    # Only use benchmark compounds for model training
    s4_benchmark = s4[s4["Dataset_Role"] == "Benchmark"].copy()

    merge_cols = ["Compound_Name", "CAS_Number", "SMILES"]
    data = pd.merge(
        s1,
        s4_benchmark,
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )

    if "ODT_mg_L" not in data.columns:
        raise ValueError("Table_S1_dataset.csv must contain 'ODT_mg_L'.")

    # Target: pT = -log10(T × 10^-3), T in mg/L
    y = -np.log10(data["ODT_mg_L"].astype(float).values * 1e-3)

    metadata_cols = {"Compound_Name", "CAS_Number", "SMILES", "Dataset_Role", "ODT_mg_L", "pT", "Skeletal_Type", "Category"}
    feature_cols = [c for c in data.columns if c not in metadata_cols]

    X_df = data[feature_cols].copy()

    # Remove high-missing features
    missing_rate = X_df.isnull().mean()
    high_missing_features = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    if high_missing_features:
        X_df = X_df.drop(columns=high_missing_features)

    # Remove zero-variance features
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_df)
    zero_variance_features = X_df.columns[~vt.get_support()].tolist()
    X_df = X_df.loc[:, vt.get_support()]

    final_feature_names = X_df.columns.tolist()

    # Impute remaining missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X_df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # XGBoost + GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    model = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="r2",
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Save model artifacts
    joblib.dump(best_model, OUTPUT_DIR / "xgboost_best_model.pkl")
    joblib.dump(imputer, OUTPUT_DIR / "imputer.pkl")
    joblib.dump(final_feature_names, OUTPUT_DIR / "final_selected_features.pkl")

    # Save metrics only
    metrics_report = {
        "n_benchmark_compounds": int(len(data)),
        "n_final_features": int(len(final_feature_names)),
        "high_missing_features_removed": int(len(high_missing_features)),
        "zero_variance_features_removed": int(len(zero_variance_features)),
        "best_params": grid_search.best_params_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    with open(OUTPUT_DIR / "xgboost_metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    print(f"Train R2: {train_metrics['R2']:.4f}")
    print(f"Test  R2: {test_metrics['R2']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Test MAE: {test_metrics['MAE']:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'xgboost_best_model.pkl'}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'xgboost_metrics_report.json'}")


if __name__ == "__main__":
    main()