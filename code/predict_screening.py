from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "Table_S3_screening_candidates.csv"
OUTPUT_FILE = OUTPUT_DIR / "screening_predictions.csv"

MODEL_PATH = OUTPUT_DIR / "xgboost_best_model.pkl"
IMPUTER_PATH = OUTPUT_DIR / "imputer.pkl"
FEATURES_PATH = OUTPUT_DIR / "final_selected_features.pkl"


def get_rdkit_descriptor_map() -> dict[str, callable]:
    return {name: func for name, func in Descriptors._descList}


def calculate_descriptors(smiles: str, descriptor_names: list[str], descriptor_map: dict[str, callable]) -> list[float]:
    if pd.isna(smiles):
        return [np.nan] * len(descriptor_names)

    mol = Chem.MolFromSmiles(str(smiles).strip())
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


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(f"Missing imputer file: {IMPUTER_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature list file: {FEATURES_PATH}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = {"Compound_Name", "CAS_Number", "SMILES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    final_feature_names = joblib.load(FEATURES_PATH)

    descriptor_map = get_rdkit_descriptor_map()

    print(f"Loaded {len(df)} candidate compounds.")
    print(f"Calculating {len(final_feature_names)} descriptors used by the trained model...")

    descriptor_rows = []
    invalid_rows = []

    for i, smiles in enumerate(df["SMILES"], start=1):
        if i % 50 == 0 or i == len(df):
            print(f"  Processed {i}/{len(df)}")
        row = calculate_descriptors(smiles, final_feature_names, descriptor_map)
        descriptor_rows.append(row)
        if all(pd.isna(v) for v in row):
            invalid_rows.append(i - 1)

    X_df = pd.DataFrame(descriptor_rows, columns=final_feature_names)
    X = imputer.transform(X_df)

    predicted_pT = model.predict(X)
    predicted_odt = 10 ** (-predicted_pT) * 1e3

    result = df[["Compound_Name", "CAS_Number", "SMILES"]].copy()
    result["Predicted_pT"] = predicted_pT
    result["Predicted_ODT_mg_L"] = predicted_odt

    if invalid_rows:
        result.loc[invalid_rows, ["Predicted_pT", "Predicted_ODT_mg_L"]] = np.nan
        print(f"Warning: {len(invalid_rows)} compounds could not be parsed from SMILES and were assigned NaN predictions.")

    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()