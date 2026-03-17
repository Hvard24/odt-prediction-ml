# Machine Learning Prediction of Aqueous Odor Detection Thresholds
This repository accompanies the study:

"Machine learning reveals structural determinants of odor detection thresholds and identifies high-potency food odorants"

It provides the datasets and code used for predicting aqueous odor detection thresholds (ODTs) of organic compounds using an optimized machine learning model.

---
## Summary

- Dataset: 1003 compounds (aqueous ODTs)
- Model: XGBoost
- Test performance: R² = 0.86
- External validation: 177 compounds, demonstrating satisfactory predictive accuracy
  
## Repository Structure

```
.
├── code/
│   ├── descriptor_calculation.py
│   ├── train_model.py
│   └── predict_screening.py
├── data/
│   ├── Table_S1_dataset.csv
│   ├── Table_S2_external_validation.csv
│   ├── Table_S3_screening_candidates.csv
│   └── Table_S4_descriptor_matrix.csv
├── README.md
└── LICENSE
```

---

## Data Description

**Table S1 – Benchmark Dataset**  
Compounds used for model training and internal validation, including SMILES and experimentally measured aqueous ODT values.

**Table S2 – External Validation Dataset**  
Independent compounds used to evaluate model generalization.

**Table S3 – Screening Candidates**  
Candidate compounds used for prospective prediction.

**Table S4 – Descriptor Matrix**  
Molecular descriptors calculated using RDKit based on canonical SMILES strings.

---

## Requirements

pip install pandas numpy scikit-learn xgboost joblib

RDKit installation (recommended via conda):

conda install -c conda-forge rdkit

---

## Usage

### 1. Descriptor Calculation

python code/descriptor_calculation.py

### 2. Model Training

python code/train_model.py

### 3. Prediction

python code/predict_screening.py

---

## Target Definition

pT = -log10(ODT × 10⁻³)

where ODT is expressed in mg/L.

---

## Reproducibility

All results can be reproduced by running the provided scripts.

Due to minor differences in descriptor calculation, preprocessing, and software environments, slight variations in model performance may occur.

---

## License

See LICENSE for details.

---

## Contact

For questions or collaboration, please contact the author.
