"""FastAPI application entry point for Clinical-Trace AI.

Builds the FastAPI app, includes the prediction/explanation routes from
``api.routes``, and configures optional LangSmith tracing on startup. This is the
process launched by ``uvicorn api.main:app`` and by the Docker image.

Safety: exposes only decision-support endpoints over synthetic patient data; it
is not a medical device and returns no medical advice.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.routes import router
from src.config import POSITIONING_STATEMENT
from src.tracing import configure_langsmith


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_langsmith()
    yield


app = FastAPI(
    title="Clinical-Trace AI",
    description=POSITIONING_STATEMENT,
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
