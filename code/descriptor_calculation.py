from __future__ import annotations

import warnings
from pathlib import Path

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

INPUT_FILES = [
    ("Table_S1_dataset.csv", "Benchmark"),
    ("Table_S2_external_validation.csv", "External_Validation"),
    ("Table_S3_screening_candidates.csv", "Screening"),
]

OUTPUT_FILE = DATA_DIR / "Table_S4_descriptor_matrix.csv"


# ============================================================
# Column name settings
# These aliases make the script compatible with slightly different
# table headers without changing the original data values.
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


# Optional columns that will be kept if they exist.
# This does not modify the values; it only carries useful metadata forward.
OPTIONAL_COLUMNS = [
    "Chemical_class",
    "Chemical_Class",
    "n_original_records",
    "ODT_min_mg_L",
    "ODT_max_mg_L",
    "ODT_geometric_mean_mg_L",
    "log10_ODT_SD",
    "Final_pT",
    "Experimental_ODT_mg_L",
    "Experimental_pT",
    "Predicted_ODT_mg_L",
    "Predicted_pT",
    "Fold_error",
    "Within_10_fold",
    "Within_100_fold",
    "Max_Tanimoto_similarity",
    "Nearest_training_compound",
    "AD_status",
    "Selection_reason",
]


def get_descriptor_names() -> list[str]:
    """
    Return all RDKit 2D descriptor names.
    """
    return [name for name, _ in Descriptors._descList]


def calculate_descriptors(smiles: str, descriptor_names: list[str]) -> list[float]:
    """
    Calculate RDKit descriptors from a SMILES string.

    If the SMILES is missing or invalid, return NaN values for all descriptors.
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
        try:
            values.append(getattr(Descriptors, name)(mol))
        except Exception:
            values.append(np.nan)

    return values


def find_column(df: pd.DataFrame, standard_name: str, filename: str) -> str:
    """
    Find the actual column name in df according to predefined aliases.
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


def load_input_table(filename: str, dataset_role: str) -> pd.DataFrame:
    """
    Load one input CSV file and standardize the key columns.

    The script only standardizes column names used by the code.
    It does not change the original data values.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    compound_col = find_column(df, "Compound_Name", filename)
    cas_col = find_column(df, "CAS_Number", filename)
    smiles_col = find_column(df, "SMILES", filename)

    # Keep key columns and standardize their names
    out = pd.DataFrame()
    out["Compound_Name"] = df[compound_col]
    out["CAS_Number"] = df[cas_col]
    out["SMILES"] = df[smiles_col]

    # Keep optional useful columns if present
    for col in OPTIONAL_COLUMNS:
        if col in df.columns and col not in out.columns:
            out[col] = df[col]

    # Add traceability information
    out["Dataset_Role"] = dataset_role
    out["Source_File"] = filename
    out["Original_Row_Index"] = df.index + 2  # +2 because CSV row 1 is header

    # Remove rows without a usable SMILES
    before = len(out)
    out = out.dropna(subset=["SMILES"]).copy()
    out["SMILES"] = out["SMILES"].astype(str).str.strip()
    out = out[out["SMILES"] != ""].copy()
    after = len(out)

    if before != after:
        print(f"{filename}: removed {before - after} empty SMILES rows.")

    return out


def main() -> None:
    descriptor_names = get_descriptor_names()

    frames = []
    for filename, dataset_role in INPUT_FILES:
        df = load_input_table(filename, dataset_role)
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    print(f"Loaded {len(all_data)} compounds with valid SMILES.")
    print(f"Calculating {len(descriptor_names)} RDKit descriptors...")

    descriptor_rows = []
    invalid_smiles_count = 0

    for i, smiles in enumerate(all_data["SMILES"], start=1):
        if i % 100 == 0 or i == len(all_data):
            print(f"  Processed {i}/{len(all_data)}")

        values = calculate_descriptors(smiles, descriptor_names)

        if all(pd.isna(v) for v in values):
            invalid_smiles_count += 1

        descriptor_rows.append(values)

    desc_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)

    output_df = pd.concat(
        [all_data.reset_index(drop=True), desc_df.reset_index(drop=True)],
        axis=1,
    )

    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved descriptor matrix to: {OUTPUT_FILE}")
    print(f"Total rows: {len(output_df)}")
    print(f"Descriptor columns: {len(descriptor_names)}")
    print(f"Invalid or unparsed SMILES rows: {invalid_smiles_count}")


if __name__ == "__main__":
    main()
