# Security Checklist

This is an educational, privacy-aware prototype using public and/or synthetic healthcare data. It is designed to demonstrate healthcare AI observability, not to provide medical advice or replace clinical judgment.

- `.env` is ignored by Git.
- `.env.example` contains placeholders only.
- Raw and processed data files are ignored.
- Model artifacts and MLflow artifacts are ignored.
- No API keys, tokens, or passwords are stored in source files.
- LLM calls receive only synthetic patient ID, risk label, risk probability, model version, and top feature names.
- Raw patient records are not logged to console, MLflow, LangSmith, FastAPI, or Streamlit.
- Explanations must include: "This is decision-support only and not medical advice."
- The LLM prompt forbids diagnosis, prescribing, guaranteed clinical outcomes, and uncited clinical literature claims.
