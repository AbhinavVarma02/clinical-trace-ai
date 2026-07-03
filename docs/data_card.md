# Data Card: UCI Diabetes 130-US Hospitals

This project uses the public UCI Diabetes 130-US Hospitals dataset for educational purposes.

## Source

UCI Machine Learning Repository: Diabetes 130-US hospitals for years 1999-2008.

## Files

- `data/raw/diabetic_data.csv`
- `data/raw/IDS_mapping.csv`

Raw and processed data files are intentionally ignored by Git.

## Target

- Positive class: `readmitted == "<30"`
- Negative class: `readmitted == ">30"` or `readmitted == "NO"`

## Required Cleaning

- Read `?` values as missing.
- Remove discharge disposition IDs `{11, 13, 14, 19, 20, 21}`.
- Split by `patient_nbr` using `GroupShuffleSplit`.
- Drop `patient_nbr` from model features after splitting.

## Risks

The dataset is historical, imbalanced, and may contain demographic or site-level biases. It is suitable for a portfolio prototype only.
