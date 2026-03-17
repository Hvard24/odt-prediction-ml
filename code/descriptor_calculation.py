from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILES = [
    ("Table_S1_dataset.csv", "Benchmark"),
    ("Table_S2_external_validation.csv", "External_Validation"),
    ("Table_S3_screening_candidates.csv", "Screening"),
]

OUTPUT_FILE = DATA_DIR / "Table_S4_descriptor_matrix.csv"


def get_descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors._descList]


def calculate_descriptors(smiles: str, descriptor_names: list[str]) -> list[float]:
    if pd.isna(smiles):
        return [np.nan] * len(descriptor_names)

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return [np.nan] * len(descriptor_names)

    values = []
    for name in descriptor_names:
        try:
            values.append(getattr(Descriptors, name)(mol))
        except Exception:
            values.append(np.nan)
    return values


def load_input_table(filename: str, dataset_role: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required_cols = {"Compound_Name", "CAS_Number", "SMILES"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")

    out = df[["Compound_Name", "CAS_Number", "SMILES"]].copy()
    out["Dataset_Role"] = dataset_role
    return out


def main() -> None:
    descriptor_names = get_descriptor_names()

    frames = []
    for filename, dataset_role in INPUT_FILES:
        df = load_input_table(filename, dataset_role)
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(all_data)} compounds.")
    print(f"Calculating {len(descriptor_names)} RDKit descriptors...")

    descriptor_rows = []
    for i, smiles in enumerate(all_data["SMILES"], start=1):
        if i % 100 == 0 or i == len(all_data):
            print(f"  Processed {i}/{len(all_data)}")
        descriptor_rows.append(calculate_descriptors(smiles, descriptor_names))

    desc_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)
    output_df = pd.concat([all_data.reset_index(drop=True), desc_df], axis=1)

    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved descriptor matrix to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()