"""Model training and prediction tests."""

from __future__ import annotations

import joblib

from src.evaluate import predict_positive_probability
from src.preprocessing import prepare_inference_features, transform_with_pipeline
from src.train import run_training


def test_training_on_sample_data_produces_model_files(tmp_path, raw_diabetes_csv):
    models_dir = tmp_path / "models"
    metadata = run_training(
        data_path=raw_diabetes_csv,
        artifact_dir=models_dir,
        processed_dir=tmp_path / "processed",
        ensure_download=False,
        quick=True,
        enable_mlflow=False,
    )

    assert (models_dir / "best_model.joblib").exists()
    assert (models_dir / "preprocessing_pipeline.joblib").exists()
    assert metadata["model_type"]
    assert "recall_positive" in metadata["metrics"]


def test_saved_model_returns_probability(tmp_path, raw_diabetes_csv):
    models_dir = tmp_path / "models"
    run_training(
        data_path=raw_diabetes_csv,
        artifact_dir=models_dir,
        processed_dir=tmp_path / "processed",
        ensure_download=False,
        quick=True,
        enable_mlflow=False,
    )
    model = joblib.load(models_dir / "best_model.joblib")
    pipeline = joblib.load(models_dir / "preprocessing_pipeline.joblib")
    sample = prepare_inference_features(
        {
            "patient_id": "synthetic_001",
            "age": "[70-80)",
            "time_in_hospital": 7,
            "num_lab_procedures": 44,
            "num_procedures": 1,
            "num_medications": 18,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 2,
            "number_diagnoses": 9,
            "insulin": "Up",
            "change": "Ch",
            "diabetesMed": "Yes",
        }
    )
    transformed = transform_with_pipeline(pipeline, sample)
    probability = float(predict_positive_probability(model, transformed)[0])
    assert 0.0 <= probability <= 1.0
