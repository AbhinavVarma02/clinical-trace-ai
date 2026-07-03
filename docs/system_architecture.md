# System Architecture

```text
data/raw/diabetic_data.csv
        |
        v
src.preprocessing
  - missing value handling
  - hospice/death filtering
  - ICD-9 category mapping
  - GroupShuffleSplit by patient_nbr
        |
        v
src.train + src.evaluate
  - Logistic Regression
  - Random Forest
  - XGBoost
  - MLflow local tracking
        |
        v
models/
  - best_model.joblib
  - preprocessing_pipeline.joblib
  - model_metadata.json
        |
        v
api/
  - /health
  - /model-info
  - /predict
  - /explain
  - /feedback
        |
        v
dashboard/app.py
```

The optional LLM path uses LangChain and LangSmith only when keys are configured. Offline demo mode uses rule-based explanations.
