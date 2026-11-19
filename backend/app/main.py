"""Main FastAPI Application Entrypoint with Lifespan Management."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.endpoints import router as api_router
from backend.app.services.database import Database
from backend.app.services.graph_store import GraphStore
from backend.app.services.inference import InferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing GraphGuard Backend services...")

    db = Database("data/inference.db")
    store = GraphStore("data/processed")
    store.load()

    engine = InferenceEngine(db=db)
    engine.load(config_path="backend/models.yaml", scalers_dir="data/processed/scalers")

    app.state.database = db
    app.state.graph_store = store
    app.state.inference_engine = engine

    logger.info("GraphGuard services initialized successfully.")
    yield
    logger.info("GraphGuard Backend shutting down...")


app = FastAPI(
    title="GraphGuard Intelligence API",
    description="Offline financial crime graph intelligence demonstration on historical Elliptic dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)

