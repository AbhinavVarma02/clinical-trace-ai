"""API routes for prediction and explanation workflows.

Defines the public HTTP surface: a safe landing page plus ``/health``,
``/model-info``, ``/predict``, ``/explain``, and ``/feedback``. All model,
SHAP, and explanation logic is delegated unchanged to ``src`` (``predict``,
``llm_explainer``, ``fallback_explainer``); this module only handles routing,
validation-error mapping, and response shaping.

Safety: only synthetic patient IDs are accepted, and every response carries the
"decision-support only" disclaimer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import HTMLResponse

from api.schemas import (
    ExplainResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src import config
from src.fallback_explainer import generate_explanation as generate_fallback_explanation
from src.llm_explainer import generate_explanation as generate_llm_explanation
from src.predict import get_model_info, is_model_loaded, predict as run_prediction


router = APIRouter()

ROOT_PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clinical-Trace AI</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f6f8fb;
      --surface: #ffffff;
      --surface-soft: #edf7f6;
      --ink: #101828;
      --muted: #344054;
      --line: #cfd8e3;
      --navy: #12315c;
      --teal: #007a74;
      --amber: #875800;
      --red: #a23b3b;
      --shadow: 0 18px 55px rgba(16, 24, 40, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      min-height: 100%;
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    body {
      padding: 32px;
    }

    a {
      color: var(--navy);
      font-weight: 700;
      text-decoration-thickness: 2px;
      text-underline-offset: 3px;
    }

    .page {
      width: min(1120px, 100%);
      margin: 0 auto;
    }

    .hero,
    .panel,
    .status-card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .hero {
      padding: clamp(28px, 5vw, 54px);
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      min-height: 32px;
      padding: 6px 12px;
      border: 1px solid #9ccfc9;
      border-radius: 999px;
      background: var(--surface-soft);
      color: #004c47;
      font-size: 0.86rem;
      font-weight: 800;
    }

    h1,
    h2,
    h3,
    p {
      margin-top: 0;
    }

    h1 {
      max-width: 920px;
      margin: 22px 0 12px;
      color: var(--ink);
      font-size: clamp(2.7rem, 7vw, 5.8rem);
      line-height: 1;
      font-weight: 860;
    }

    h2 {
      color: var(--ink);
      font-size: clamp(1.25rem, 2vw, 1.7rem);
      line-height: 1.2;
      margin-bottom: 12px;
    }

    h3 {
      color: var(--ink);
      font-size: 1rem;
      line-height: 1.35;
      margin-bottom: 8px;
    }

    p,
    li,
    small {
      color: var(--muted);
      line-height: 1.6;
    }

    .subtitle {
      max-width: 850px;
      color: #1d2939;
      font-size: clamp(1.08rem, 2vw, 1.38rem);
      font-weight: 650;
    }

    .disclaimer {
      width: min(820px, 100%);
      margin: 24px 0 0;
      padding: 14px 16px;
      border-left: 5px solid var(--amber);
      border-radius: 6px;
      background: #fff7e6;
      color: #513500;
      font-weight: 750;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 28px;
    }

    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 12px 18px;
      border: 1px solid var(--navy);
      border-radius: 6px;
      background: #ffffff;
      color: var(--navy);
      font-size: 0.98rem;
      font-weight: 800;
      text-decoration: none;
    }

    .button.primary {
      background: var(--navy);
      color: #ffffff;
    }

    .button.secondary {
      border-color: var(--teal);
      color: #005f5a;
    }

    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }

    .status-card {
      min-height: 142px;
      padding: 18px;
      box-shadow: none;
    }

    .status-card span {
      display: block;
      color: #475467;
      font-size: 0.82rem;
      font-weight: 800;
      text-transform: uppercase;
    }

    .status-card strong {
      display: block;
      margin: 10px 0 6px;
      color: var(--ink);
      font-size: 1.22rem;
      line-height: 1.25;
    }

    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
      gap: 18px;
      align-items: stretch;
    }

    .panel {
      padding: 24px;
      box-shadow: none;
    }

    .command {
      display: block;
      width: 100%;
      margin: 16px 0 14px;
      padding: 14px;
      overflow-wrap: anywhere;
      border: 1px solid #b8c4d3;
      border-radius: 6px;
      background: #f8fafc;
      color: #182230;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 0.92rem;
      line-height: 1.45;
    }

    .endpoint-list,
    .safety-list {
      display: grid;
      gap: 10px;
      padding: 0;
      margin: 14px 0 0;
      list-style: none;
    }

    .endpoint-list li,
    .safety-list li {
      padding: 12px 14px;
      border: 1px solid #d5dde8;
      border-radius: 6px;
      background: #fbfcfe;
      color: #243447;
      font-weight: 650;
    }

    .method {
      display: inline-block;
      min-width: 52px;
      margin-right: 10px;
      color: #005f5a;
      font-weight: 900;
    }

    .note {
      margin-top: 16px;
      color: #3c4856;
      font-size: 0.95rem;
    }

    .safety-strip {
      margin-top: 18px;
      border-top: 4px solid var(--teal);
    }

    .safety-strip h2 {
      margin-bottom: 6px;
    }

    .danger {
      color: var(--red);
      font-weight: 800;
    }

    @media (max-width: 860px) {
      body {
        padding: 18px;
      }

      .status-grid,
      .content-grid {
        grid-template-columns: 1fr;
      }

      .hero {
        padding: 26px;
      }

      h1 {
        font-size: clamp(2.35rem, 13vw, 3.8rem);
        line-height: 1.04;
      }

      .button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main class="page" aria-label="Clinical-Trace AI service home">
    <header class="hero">
      <div class="eyebrow">Healthcare AI Observability</div>
      <h1>Clinical-Trace AI</h1>
      <p class="subtitle">Healthcare Readmission Risk Platform with MLOps &amp; LLMOps Observability</p>
      <p class="disclaimer">__SAFETY_DISCLAIMER__</p>
      <nav class="actions" aria-label="Primary service links">
        <a class="button primary" href="/docs">Open API Docs</a>
        <a class="button secondary" href="/health">View Health</a>
        <a class="button" href="/model-info">Model Info</a>
      </nav>
    </header>

    <section class="status-grid" aria-label="System overview">
      <article class="status-card">
        <span>Interface</span>
        <strong>FastAPI Service</strong>
        <small>Stable backend endpoints for prediction, explanation, health, and metadata.</small>
      </article>
      <article class="status-card">
        <span>Dashboard</span>
        <strong>Streamlit Product UI</strong>
        <small>Run the dashboard on port 8501 for the full simulator and observability panels.</small>
      </article>
      <article class="status-card">
        <span>Privacy</span>
        <strong>Synthetic IDs Only</strong>
        <small>No PHI is shown here, and raw patient records are not sent to LLM providers.</small>
      </article>
      <article class="status-card">
        <span>Fallback</span>
        <strong>Offline Demo Ready</strong>
        <small>Rule-based explanations work when OpenAI or LangSmith keys are not configured.</small>
      </article>
    </section>

    <section class="content-grid" aria-label="Product entry points">
      <article class="panel">
        <h2>Product Dashboard</h2>
        <p>Use the Streamlit dashboard for patient risk simulation, risk cards, feature contributions, explanations, model performance, recent predictions, and safety controls.</p>
        <code class="command">streamlit run dashboard/app.py --server.port 8501</code>
        <a class="button secondary" href="http://127.0.0.1:8501">Open Streamlit Dashboard</a>
        <p class="note">The dashboard is designed for synthetic demo encounters and keeps the clinical disclaimer visible.</p>
      </article>

      <article class="panel">
        <h2>API Surface</h2>
        <ul class="endpoint-list">
          <li><span class="method">GET</span><a href="/health">/health</a></li>
          <li><span class="method">GET</span><a href="/model-info">/model-info</a></li>
          <li><span class="method">POST</span>/predict</li>
          <li><span class="method">POST</span>/explain</li>
          <li><span class="method">POST</span>/feedback</li>
        </ul>
        <p class="note">Use <a href="/docs">/docs</a> for interactive request schemas and test calls.</p>
      </article>
    </section>

    <section class="panel safety-strip" aria-label="Safety and privacy">
      <h2>Safety And Privacy</h2>
      <p>No raw clinical records, secrets, or PHI are displayed on this landing page.</p>
      <ul class="safety-list">
        <li>No PHI used in the interface.</li>
        <li>Synthetic patient IDs only for demos and tests.</li>
        <li>No raw patient records sent to LLM providers.</li>
        <li>Offline demo mode works without OpenAI or LangSmith keys.</li>
      </ul>
      <p class="note"><span class="danger">Decision support only:</span> always use clinical judgment and the full patient context.</p>
    </section>
  </main>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Return a safe product-style landing page for browser visits."""
    return HTMLResponse(
        content=ROOT_PAGE_HTML.replace("__SAFETY_DISCLAIMER__", config.SAFETY_DISCLAIMER)
    )
@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid noisy 404s from browser favicon requests."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and mode flags."""
    return HealthResponse(
        model_loaded=is_model_loaded(),
        llm_mode_active=config.LLM_AVAILABLE,
        langsmith_active=config.LANGSMITH_AVAILABLE,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return deployed model metadata."""
    metadata = get_model_info()
    return ModelInfoResponse(
        model_type=str(metadata.get("model_type", "untrained")),
        model_version=str(metadata.get("model_version", "local")),
        training_date=str(metadata.get("training_date", "not trained")),
        metrics=dict(metadata.get("metrics", {})),
        risk_threshold=float(metadata.get("risk_threshold", config.RISK_THRESHOLD)),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run the readmission risk model."""
    request_id = str(uuid4())
    try:
        prediction = run_prediction(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return PredictResponse(request_id=request_id, **prediction)


@router.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest) -> ExplainResponse:
    """Run prediction and produce a safe explanation."""
    request_id = str(uuid4())
    try:
        prediction = run_prediction(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if config.LLM_AVAILABLE:
        explanation = generate_llm_explanation(
            patient_id=prediction["patient_id"],
            risk_label=prediction["readmission_risk"],
            risk_probability=prediction["risk_probability"],
            top_features=prediction["top_features"],
            request_id=request_id,
            model_version=prediction["model_version"],
        )
    else:
        explanation = generate_fallback_explanation(
            patient_id=prediction["patient_id"],
            risk_label=prediction["readmission_risk"],
            risk_probability=prediction["risk_probability"],
            top_features=prediction["top_features"],
            request_id=request_id,
            model_version=prediction["model_version"],
        )
    return ExplainResponse(**explanation)


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Accept MVP feedback without persistent storage."""
    return FeedbackResponse(request_id=request.request_id)
