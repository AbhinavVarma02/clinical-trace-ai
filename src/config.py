"""Application configuration and environment loading for Clinical-Trace AI.

Central place for filesystem paths (data, models, prompts, MLflow), the feature
flags derived from environment variables (LLM / LangSmith availability), the
risk threshold, and the shared safety/positioning statements. Imported by nearly
every other module so paths and modes stay consistent across training, the API,
and the dashboard.

Safety: optional keys are read from a local ``.env`` (never committed) and only
booleans such as ``LLM_AVAILABLE`` are re-exported — no secret values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.getenv("CLINICAL_TRACE_SKIP_DOTENV", "").lower() not in {"1", "true", "yes"}:
    load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DOCS_DIR = PROJECT_ROOT / "docs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_DATA_PATH = RAW_DATA_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = RAW_DATA_DIR / "IDS_mapping.csv"
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/best_model.joblib")
PIPELINE_PATH = PROJECT_ROOT / os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "clinical-trace-ai")

LLM_AVAILABLE = bool(OPENAI_API_KEY)
LANGSMITH_AVAILABLE = bool(LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2)

RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.35"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local")

POSITIONING_STATEMENT = (
    "This is an educational, privacy-aware prototype using public and/or synthetic "
    "healthcare data. It is designed to demonstrate healthcare AI observability, "
    "not to provide medical advice or replace clinical judgment."
)
SAFETY_DISCLAIMER = "This is decision-support only and not medical advice."


def ensure_directories() -> None:
    """Create local artifact directories used by the app."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        PROMPTS_DIR,
        MLRUNS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_model_metadata() -> dict[str, Any]:
    """Load model metadata if training has produced it."""
    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "model_type": "untrained",
        "model_version": MODEL_VERSION,
        "training_date": "not trained",
        "metrics": {},
        "risk_threshold": RISK_THRESHOLD,
    }


def configured_risk_threshold() -> float:
    """Return the deployed threshold from metadata, falling back to env/default."""
    metadata = load_model_metadata()
    try:
        return float(metadata.get("risk_threshold", RISK_THRESHOLD))
    except (TypeError, ValueError):
        return RISK_THRESHOLD
