from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# Global settings
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
MISSING_THRESHOLD = 0.10
N_Y_RANDOMIZATION = 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S1_FILE = DATA_DIR / "Table_S1_dataset.csv"
S4_FILE = DATA_DIR / "Table_S4_descriptor_matrix.csv"


# ============================================================
# Column aliases
# This keeps the code compatible with slightly different table headers.
# It does not change original data values.
# ============================================================
COLUMN_ALIASES = {
    "Compound_Name": [
        "Compound_Name",
        "Compound_name",
        "Compound name",
        "Compound",
        "Name",
        "compound_name",
        "compound",
    ],
    "CAS_Number": [
        "CAS_Number",
        "CAS",
        "CAS number",
        "CAS_No",
        "CAS No.",
        "cas",
    ],
    "SMILES": [
        "SMILES",
        "Canonical_SMILES",
        "Canonical SMILES",
        "canonical_smiles",
        "CanonicalSmiles",
        "Smiles",
    ],
    "Final_pT": [
        "Final_pT",
        "pT",
        "final_pT",
        "Target_pT",
        "Experimental_pT",
    ],
    "ODT_mg_L": [
        "ODT_mg_L",
        "ODT_geometric_mean_mg_L",
        "Geometric_mean_ODT_mg_L",
        "Final_ODT_mg_L",
        "Experimental_ODT_mg_L",
        "ODT",
    ],
}


def find_column(
    df: pd.DataFrame,
    standard_name: str,
    filename: str,
    required: bool = True,
) -> str | None:
    """
    Find an actual column name from predefined aliases.
    """
    aliases = COLUMN_ALIASES[standard_name]

    for col in aliases:
        if col in df.columns:
            return col

    lower_map = {str(col).lower().strip(): col for col in df.columns}
    for col in aliases:
        key = col.lower().strip()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise ValueError(
            f"{filename} is missing required column for '{standard_name}'. "
            f"Accepted names: {aliases}. "
            f"Current columns: {list(df.columns)}"
        )

    return None


def standardize_key_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Standardize key identifiers to:
    Compound_Name, CAS_Number, SMILES

    Only column names are standardized. Data values are not modified.
    """
    out = df.copy()

    compound_col = find_column(out, "Compound_Name", filename, required=True)
    cas_col = find_column(out, "CAS_Number", filename, required=True)
    smiles_col = find_column(out, "SMILES", filename, required=True)

    if compound_col != "Compound_Name":
        out["Compound_Name"] = out[compound_col]
    if cas_col != "CAS_Number":
        out["CAS_Number"] = out[cas_col]
    if smiles_col != "SMILES":
        out["SMILES"] = out[smiles_col]

    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def pT_from_odt_mg_L(odt_mg_L: pd.Series | np.ndarray) -> np.ndarray:
    """
    Manuscript definition:
        pT = -log10(ODT_mg_L × 10^-3)
    """
    odt = pd.to_numeric(odt_mg_L, errors="coerce").astype(float)
    odt = np.clip(odt, 1e-30, None)
    return -np.log10(odt * 1e-3)


def prepare_target(s1: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare modeling target from Table S1.

    Priority:
    1. Use Final_pT if available.
    2. Otherwise compute pT from ODT_mg_L / ODT_geometric_mean_mg_L.
    """
    s1 = s1.copy()

    pt_col = find_column(s1, "Final_pT", S1_FILE.name, required=False)
    odt_col = find_column(s1, "ODT_mg_L", S1_FILE.name, required=False)

    if pt_col is not None:
        s1["Target_pT"] = pd.to_numeric(s1[pt_col], errors="coerce")
        target_source = pt_col
    elif odt_col is not None:
        s1["Target_pT"] = pT_from_odt_mg_L(s1[odt_col])
        target_source = odt_col
    else:
        raise ValueError(
            "Table_S1_dataset.csv must contain either a pT column "
            "such as 'Final_pT' or an ODT column such as "
            "'ODT_geometric_mean_mg_L' / 'ODT_mg_L'."
        )

    s1 = s1.dropna(subset=["Target_pT"]).copy()
    print(f"Target column source: {target_source}")
    return s1


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated column names after merging.
    """
    return df.loc[:, ~df.columns.duplicated()].copy()


def get_feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Extract numeric descriptor features and remove non-feature metadata columns.
    """
    metadata_cols = {
        # identifiers
        "Compound_Name",
        "CAS_Number",
        "SMILES",
        "Canonical_SMILES",
        "Compound_name",
        "CAS",
        # dataset labels
        "Dataset_Role",
        "Source_File",
        "Original_Row_Index",
        # target and ODT information
        "Target_pT",
        "Final_pT",
        "pT",
        "ODT_mg_L",
        "ODT_geometric_mean_mg_L",
        "Geometric_mean_ODT_mg_L",
        "Final_ODT_mg_L",
        "Experimental_ODT_mg_L",
        "Experimental_pT",
        # descriptive information
        "Chemical_class",
        "Chemical_Class",
        "Skeletal_Type",
        "Category",
        "Source_IDs",
        "n_original_records",
        "ODT_min_mg_L",
        "ODT_max_mg_L",
        "log10_ODT_SD",
        # prediction / validation result columns
        "Predicted_pT",
        "Predicted_ODT_mg_L",
        "Absolute_error_pT",
        "Fold_error",
        "Within_10_fold",
        "Within_100_fold",
        "Max_Tanimoto_similarity",
        "Nearest_training_compound",
        "AD_status",
        "Selection_reason",
    }

    candidate_cols = [c for c in data.columns if c not in metadata_cols]

    # Convert possible descriptor columns to numeric.
    X_candidate = data[candidate_cols].apply(pd.to_numeric, errors="coerce")

    # Keep only columns that contain at least one numeric value.
    numeric_cols = X_candidate.columns[X_candidate.notna().any()].tolist()
    X_df = X_candidate[numeric_cols].copy()

    feature_info = {
        "n_candidate_feature_columns": int(len(candidate_cols)),
        "n_numeric_feature_columns": int(len(numeric_cols)),
        "non_numeric_or_empty_features_removed": int(len(candidate_cols) - len(numeric_cols)),
    }

    return X_df, feature_info


def summarize_cv_scores(cv_result: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Summarize 5-fold CV scores.
    """
    r2_scores = cv_result["test_R2"]
    rmse_scores = -cv_result["test_RMSE"]
    mae_scores = -cv_result["test_MAE"]

    return {
        "CV_R2_mean": float(np.mean(r2_scores)),
        "CV_R2_SD": float(np.std(r2_scores, ddof=1)),
        "CV_RMSE_mean": float(np.mean(rmse_scores)),
        "CV_RMSE_SD": float(np.std(rmse_scores, ddof=1)),
        "CV_MAE_mean": float(np.mean(mae_scores)),
        "CV_MAE_SD": float(np.std(mae_scores, ddof=1)),
    }


def run_y_randomization(
    best_model: xgb.XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    True Y-randomization:
    The training labels are randomly permuted, the model is refit,
    and performance is evaluated on the original test labels.

    Low test R2 after randomization supports that the original model
    did not learn from random label structure.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for i in range(n_iter):
        y_train_perm = rng.permutation(y_train)

        model_perm = clone(best_model)
        model_perm.set_params(random_state=random_state + i + 1)
        model_perm.fit(X_train, y_train_perm)

        y_train_perm_pred = model_perm.predict(X_train)
        y_test_perm_pred = model_perm.predict(X_test)

        train_metrics = compute_metrics(y_train_perm, y_train_perm_pred)
        test_metrics = compute_metrics(y_test, y_test_perm_pred)

        rows.append(
            {
                "iteration": i + 1,
                "train_R2_permuted_y": train_metrics["R2"],
                "train_RMSE_permuted_y": train_metrics["RMSE"],
                "train_MAE_permuted_y": train_metrics["MAE"],
                "test_R2_original_y": test_metrics["R2"],
                "test_RMSE_original_y": test_metrics["RMSE"],
                "test_MAE_original_y": test_metrics["MAE"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    # ========================================================
    # Load data
    # ========================================================
    if not S1_FILE.exists():
        raise FileNotFoundError(f"Missing file: {S1_FILE}")
    if not S4_FILE.exists():
        raise FileNotFoundError(f"Missing file: {S4_FILE}")

    s1 = pd.read_csv(S1_FILE, encoding="utf-8-sig")
    s4 = pd.read_csv(S4_FILE, encoding="utf-8-sig")

    s1 = standardize_key_columns(s1, S1_FILE.name)
    s4 = standardize_key_columns(s4, S4_FILE.name)

    s1 = prepare_target(s1)

    # Only use benchmark compounds for model training
    if "Dataset_Role" in s4.columns:
        s4_benchmark = s4[s4["Dataset_Role"] == "Benchmark"].copy()
    else:
        print("Warning: Dataset_Role column not found in S4. Using all rows in S4.")
        s4_benchmark = s4.copy()

    merge_cols = ["Compound_Name", "CAS_Number", "SMILES"]

    data = pd.merge(
        s1,
        s4_benchmark,
        on=merge_cols,
        how="inner",
        validate="one_to_one",
        suffixes=("_S1", "_S4"),
    )

    data = remove_duplicate_columns(data)

    if len(data) == 0:
        raise ValueError(
            "No matched benchmark compounds after merging S1 and S4. "
            "Please check Compound_Name, CAS_Number, and SMILES consistency."
        )

    print(f"Merged benchmark compounds: {len(data)}")

    y = data["Target_pT"].astype(float).values

    # ========================================================
    # Feature extraction and filtering
    # ========================================================
    X_df, feature_info = get_feature_matrix(data)

    if X_df.shape[1] == 0:
        raise ValueError("No numeric descriptor features found for model training.")

    print(f"Initial numeric feature columns: {X_df.shape[1]}")

    # Remove high-missing features
    missing_rate = X_df.isnull().mean()
    high_missing_features = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()

    if high_missing_features:
        X_df = X_df.drop(columns=high_missing_features)

    print(f"Removed high-missing features: {len(high_missing_features)}")

    # Remove zero-variance features
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X_df)

    zero_variance_features = X_df.columns[~vt.get_support()].tolist()
    X_df = X_df.loc[:, vt.get_support()].copy()

    final_feature_names = X_df.columns.tolist()

    print(f"Removed zero-variance features: {len(zero_variance_features)}")
    print(f"Final selected features before imputation: {len(final_feature_names)}")

    # Impute remaining missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X_df)

    # ========================================================
    # Train/test split
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # ========================================================
    # XGBoost + GridSearchCV
    # ========================================================
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.01],
        "reg_lambda": [1, 5],
    }

    base_model = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="r2",
        verbose=0,
        return_train_score=True,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # ========================================================
    # Train/test metrics
    # ========================================================
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # ========================================================
    # 5-fold CV metrics using the best model
    # ========================================================
    print("Running 5-fold cross-validation for final model...")

    scoring = {
        "R2": "r2",
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
    }

    cv_result = cross_validate(
        best_model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_metrics = summarize_cv_scores(cv_result)

    # ========================================================
    # True Y-randomization
    # ========================================================
    print(f"Running Y-randomization ({N_Y_RANDOMIZATION} iterations)...")

    y_randomization_df = run_y_randomization(
        best_model=best_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=N_Y_RANDOMIZATION,
        random_state=RANDOM_STATE,
    )

    y_randomization_summary = {
        "n_iterations": int(N_Y_RANDOMIZATION),
        "test_R2_original_y_mean": float(y_randomization_df["test_R2_original_y"].mean()),
        "test_R2_original_y_SD": float(y_randomization_df["test_R2_original_y"].std(ddof=1)),
        "test_RMSE_original_y_mean": float(y_randomization_df["test_RMSE_original_y"].mean()),
        "test_RMSE_original_y_SD": float(y_randomization_df["test_RMSE_original_y"].std(ddof=1)),
        "test_MAE_original_y_mean": float(y_randomization_df["test_MAE_original_y"].mean()),
        "test_MAE_original_y_SD": float(y_randomization_df["test_MAE_original_y"].std(ddof=1)),
    }

    # ========================================================
    # Save model artifacts
    # ========================================================
    joblib.dump(best_model, OUTPUT_DIR / "xgboost_best_model.pkl")
    joblib.dump(imputer, OUTPUT_DIR / "imputer.pkl")
    joblib.dump(final_feature_names, OUTPUT_DIR / "final_selected_features.pkl")

    # Save feature filtering details
    pd.DataFrame({"removed_high_missing_features": high_missing_features}).to_csv(
        OUTPUT_DIR / "removed_high_missing_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"removed_zero_variance_features": zero_variance_features}).to_csv(
        OUTPUT_DIR / "removed_zero_variance_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"final_selected_features": final_feature_names}).to_csv(
        OUTPUT_DIR / "final_selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(grid_search.cv_results_).to_csv(
        OUTPUT_DIR / "xgboost_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y_randomization_df.to_csv(
        OUTPUT_DIR / "y_randomization_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save train/test predictions
    pd.DataFrame(
        {
            "Observed_pT": y_train,
            "Predicted_pT": y_train_pred,
            "Dataset": "Train",
        }
    ).to_csv(
        OUTPUT_DIR / "train_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Observed_pT": y_test,
            "Predicted_pT": y_test_pred,
            "Dataset": "Test",
        }
    ).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save metrics report
    metrics_report = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "missing_threshold": MISSING_THRESHOLD,
        "n_benchmark_compounds": int(len(data)),
        "n_initial_numeric_features": int(feature_info["n_numeric_feature_columns"]),
        "n_final_features": int(len(final_feature_names)),
        "feature_info": feature_info,
        "high_missing_features_removed": int(len(high_missing_features)),
        "zero_variance_features_removed": int(len(zero_variance_features)),
        "param_grid": param_grid,
        "best_params": grid_search.best_params_,
        "best_cv_score_in_grid_search_R2": float(grid_search.best_score_),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "five_fold_cv_metrics": cv_metrics,
        "y_randomization_summary": y_randomization_summary,
    }

    with open(OUTPUT_DIR / "xgboost_metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    # ========================================================
    # Print summary
    # ========================================================
    print("\nTraining complete.")
    print(f"Benchmark compounds: {len(data)}")
    print(f"Final features: {len(final_feature_names)}")
    print(f"Best params: {grid_search.best_params_}")

    print("\nTrain metrics")
    print(f"Train R2   : {train_metrics['R2']:.4f}")
    print(f"Train RMSE : {train_metrics['RMSE']:.4f}")
    print(f"Train MAE  : {train_metrics['MAE']:.4f}")

    print("\nTest metrics")
    print(f"Test R2    : {test_metrics['R2']:.4f}")
    print(f"Test RMSE  : {test_metrics['RMSE']:.4f}")
    print(f"Test MAE   : {test_metrics['MAE']:.4f}")

    print("\n5-fold CV metrics")
    print(f"CV R2      : {cv_metrics['CV_R2_mean']:.4f} ± {cv_metrics['CV_R2_SD']:.4f}")
    print(f"CV RMSE    : {cv_metrics['CV_RMSE_mean']:.4f} ± {cv_metrics['CV_RMSE_SD']:.4f}")
    print(f"CV MAE     : {cv_metrics['CV_MAE_mean']:.4f} ± {cv_metrics['CV_MAE_SD']:.4f}")

    print("\nY-randomization summary")
    print(
        "Randomized test R2: "
        f"{y_randomization_summary['test_R2_original_y_mean']:.4f} "
        f"± {y_randomization_summary['test_R2_original_y_SD']:.4f}"
    )

    print(f"\nModel saved to: {OUTPUT_DIR / 'xgboost_best_model.pkl'}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'xgboost_metrics_report.json'}")


if __name__ == "__main__":
    main()
