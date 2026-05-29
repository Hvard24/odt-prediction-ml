from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings("ignore")

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "Table_S3_screening_candidates.csv"
OUTPUT_FILE = OUTPUT_DIR / "screening_predictions.csv"

MODEL_PATH = OUTPUT_DIR / "xgboost_best_model.pkl"
IMPUTER_PATH = OUTPUT_DIR / "imputer.pkl"
FEATURES_PATH = OUTPUT_DIR / "final_selected_features.pkl"


# ============================================================
# Column aliases
# This allows the script to work with slightly different table headers.
# It does not modify original data values.
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
        "CAS_Number",
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
}


def find_column(df: pd.DataFrame, standard_name: str, filename: str) -> str:
    """
    Find the actual column name according to predefined aliases.
    """
    aliases = COLUMN_ALIASES[standard_name]

    # Direct matching
    for col in aliases:
        if col in df.columns:
            return col

    # Case-insensitive matching
    lower_map = {str(col).lower().strip(): col for col in df.columns}
    for col in aliases:
        key = col.lower().strip()
        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"{filename} is missing required column for '{standard_name}'. "
        f"Accepted names: {aliases}. "
        f"Current columns: {list(df.columns)}"
    )


def get_rdkit_descriptor_map() -> dict[str, Callable]:
    """
    Return a mapping from RDKit descriptor name to descriptor function.
    """
    return {name: func for name, func in Descriptors._descList}


def calculate_descriptors(
    smiles: str,
    descriptor_names: list[str],
    descriptor_map: dict[str, Callable],
) -> list[float]:
    """
    Calculate selected RDKit descriptors for one SMILES string.

    If SMILES is missing or invalid, return NaN values for all descriptors.
    """
    if pd.isna(smiles):
        return [np.nan] * len(descriptor_names)

    smiles = str(smiles).strip()
    if smiles == "":
        return [np.nan] * len(descriptor_names)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)

    values = []

    for name in descriptor_names:
        func = descriptor_map.get(name)

        if func is None:
            values.append(np.nan)
            continue

        try:
            values.append(func(mol))
        except Exception:
            values.append(np.nan)

    return values


def pT_to_odt_mg_L(predicted_pT: np.ndarray) -> np.ndarray:
    """
    Convert predicted pT back to ODT in mg/L.

    Manuscript definition:
        pT = -log10(ODT_mg_L × 10^-3)

    Therefore:
        ODT_mg_L = 10^(3 - pT)
    """
    return 10 ** (3 - predicted_pT)


def main() -> None:
    # ========================================================
    # Check required files
    # ========================================================
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(f"Missing imputer file: {IMPUTER_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature list file: {FEATURES_PATH}")

    # ========================================================
    # Load candidate table
    # ========================================================
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    compound_col = find_column(df, "Compound_Name", INPUT_FILE.name)
    cas_col = find_column(df, "CAS_Number", INPUT_FILE.name)
    smiles_col = find_column(df, "SMILES", INPUT_FILE.name)

    # Keep a clean standardized copy for output
    standardized = df.copy()
    standardized["Compound_Name"] = df[compound_col]
    standardized["CAS_Number"] = df[cas_col]
    standardized["SMILES"] = df[smiles_col]

    # Remove fully empty rows
    before_rows = len(standardized)
    standardized = standardized.dropna(how="all").copy()
    after_rows = len(standardized)

    if before_rows != after_rows:
        print(f"Removed {before_rows - after_rows} fully empty rows.")

    # ========================================================
    # Load trained model components
    # ========================================================
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    final_feature_names = joblib.load(FEATURES_PATH)

    if not isinstance(final_feature_names, list):
        final_feature_names = list(final_feature_names)

    descriptor_map = get_rdkit_descriptor_map()

    # Check whether all model features exist in RDKit descriptors
    missing_descriptor_features = [
        feature for feature in final_feature_names if feature not in descriptor_map
    ]

    if missing_descriptor_features:
        print("Warning: Some selected model features are not RDKit descriptors:")
        for feature in missing_descriptor_features:
            print(f"  - {feature}")
        print("These features will be filled with NaN and handled by the saved imputer.")

    print(f"Loaded {len(standardized)} candidate compounds.")
    print(f"Calculating {len(final_feature_names)} descriptors used by the trained model...")

    # ========================================================
    # Calculate descriptors
    # ========================================================
    descriptor_rows = []
    invalid_rows = []

    for i, smiles in enumerate(standardized["SMILES"], start=1):
        if i % 50 == 0 or i == len(standardized):
            print(f"  Processed {i}/{len(standardized)}")

        row = calculate_descriptors(smiles, final_feature_names, descriptor_map)
        descriptor_rows.append(row)

        if all(pd.isna(v) for v in row):
            invalid_rows.append(i - 1)

    X_df = pd.DataFrame(descriptor_rows, columns=final_feature_names)

    # Ensure feature order is exactly the same as training
    X_df = X_df[final_feature_names]

    # ========================================================
    # Impute and predict
    # ========================================================
    X = imputer.transform(X_df)

    predicted_pT = model.predict(X)
    predicted_odt = pT_to_odt_mg_L(predicted_pT)

    # ========================================================
    # Build output table
    # ========================================================
    result = standardized.copy()

    # Avoid duplicate old prediction columns causing confusion
    for col in ["Predicted_pT", "Predicted_ODT_mg_L"]:
        if col in result.columns:
            result = result.drop(columns=[col])

    result["Predicted_pT"] = predicted_pT
    result["Predicted_ODT_mg_L"] = predicted_odt

    if invalid_rows:
        result.loc[invalid_rows, ["Predicted_pT", "Predicted_ODT_mg_L"]] = np.nan
        print(
            f"Warning: {len(invalid_rows)} compounds could not be parsed from SMILES "
            f"and were assigned NaN predictions."
        )

    # Put important columns first
    front_cols = ["Compound_Name", "CAS_Number", "SMILES", "Predicted_pT", "Predicted_ODT_mg_L"]
    other_cols = [col for col in result.columns if col not in front_cols]
    result = result[front_cols + other_cols]

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved predictions to: {OUTPUT_FILE}")
    print(f"Total candidates: {len(result)}")
    print(f"Invalid or unparsed SMILES rows: {len(invalid_rows)}")


if __name__ == "__main__":
    main()
