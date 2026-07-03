"""LangSmith tracing helpers with privacy-preserving metadata.

Detects whether LangSmith tracing is configured and builds the metadata dict
attached to LLM calls in ``src.llm_explainer``. Tracing is entirely optional and
off by default.

Safety: the metadata contains only safe fields (request ID, synthetic patient
ID, model version, risk score/label, prompt version, top feature names, safety
flags) — never raw patient values or secrets.
"""

from __future__ import annotations

from typing import Any

from src import config


def configure_langsmith() -> bool:
    """Return whether LangSmith tracing is configured."""
    if config.LANGSMITH_AVAILABLE:
        return True
    print("LangSmith tracing is not configured. Running without tracing.")
    return False


def safe_trace_metadata(
    request_id: str,
    patient_id: str,
    model_version: str,
    risk_score: float,
    risk_label: str,
    prompt_version: str,
    top_feature_names: list[str],
    disclaimer_present: bool,
    no_treatment_recommendation: bool,
) -> dict[str, Any]:
    """Build trace metadata without raw patient values or secrets."""
    return {
        "request_id": request_id,
        "synthetic_patient_id": patient_id,
        "model_version": model_version,
        "risk_score": float(risk_score),
        "risk_label": risk_label,
        "prompt_version": prompt_version,
        "top_feature_names": top_feature_names[:5],
        "safety_flags": {
            "disclaimer_present": disclaimer_present,
            "no_treatment_recommendation": no_treatment_recommendation,
        },
    }
