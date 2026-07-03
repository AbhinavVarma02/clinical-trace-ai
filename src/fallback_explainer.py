"""Rule-based offline explanation generator.

Produces a safe, deterministic explanation from model outputs alone, used
whenever the LLM layer is unavailable or its output fails validation. This is
what powers the zero-key "offline demo mode".

Safety: emits only templated language derived from the risk label, probability,
and top feature names, and always appends the decision-support disclaimer.
"""

from __future__ import annotations

from typing import Any

from src.config import SAFETY_DISCLAIMER


def _feature_name(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("feature", "model feature"))
    if isinstance(feature, (list, tuple)) and feature:
        return str(feature[0])
    return str(getattr(feature, "feature", "model feature"))


def _risk_drivers(top_features: list[Any]) -> list[str]:
    return [_feature_name(feature) for feature in top_features[:5]]


def _review_areas(drivers: list[str]) -> list[str]:
    areas: list[str] = []
    joined = " ".join(drivers).lower()
    if "inpatient" in joined or "outpatient" in joined or "emergency" in joined:
        areas.append("Recent utilization patterns")
    if "medication" in joined or "insulin" in joined or "metformin" in joined:
        areas.append("Medication complexity signals")
    if "time in hospital" in joined or "diagnosis" in joined:
        areas.append("Encounter context and diagnosis signals")
    if not areas:
        areas.append("Top model-ranked factors")
    return areas[:3]


def generate_explanation(
    patient_id: str,
    risk_label: str,
    risk_probability: float,
    top_features: list[Any],
    request_id: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Generate a safe template explanation using model outputs only."""
    drivers = _risk_drivers(top_features)
    driver_text = ", ".join(drivers) if drivers else "the available model features"
    areas = _review_areas(drivers)
    explanation = (
        f"The model's analysis flags synthetic patient {patient_id} as {risk_label} risk "
        f"with an estimated {risk_probability:.0%} probability of 30-day readmission. "
        f"The model's top contributing factors include {driver_text}. "
        f"The care team may review {', '.join(area.lower() for area in areas)} alongside "
        f"their normal workflow. {SAFETY_DISCLAIMER}"
    )
    return {
        "request_id": request_id,
        "patient_id": patient_id,
        "risk_label": risk_label,
        "risk_probability": float(risk_probability),
        "explanation": explanation,
        "risk_drivers": drivers,
        "suggested_review_areas": areas,
        "safety_disclaimer": SAFETY_DISCLAIMER,
        "prompt_version": "rule-based",
        "explanation_mode": "rule-based",
        "trace_id": None,
        "model_version": model_version,
    }
