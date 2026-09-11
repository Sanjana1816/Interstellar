"""
FastAPI application entry point for the Interstellar system.

Initializes the database, loads the embedding model, and mounts all routers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.services import embedding, vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("  INTERSTELLAR — Starting up")
    logger.info("=" * 60)

    # Ensure directories
    settings.ensure_directories()

    # Initialize database
    init_db()
    logger.info("Database initialized.")

    # Load embedding model
    try:
        embedding.load_model()
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.warning(f"Could not load embedding model: {e}")
        logger.warning("Semantic search will not be available until the model is loaded.")

    # Initialize FAISS index
    vector_store.initialize()
    logger.info(f"FAISS index ready: {vector_store.get_index_size()} vectors")

    # Mount static files for thumbnails
    thumbnails_dir = settings.data_thumbnails_dir
    if thumbnails_dir.exists():
        app.mount(
            "/thumbnails",
            StaticFiles(directory=str(thumbnails_dir)),
            name="thumbnails",
        )

    logger.info("=" * 60)
    logger.info("  INTERSTELLAR — Ready")
    logger.info(f"  API: http://localhost:{settings.api_port}")
    logger.info(f"  Docs: http://localhost:{settings.api_port}/docs")
    logger.info("=" * 60)

    yield

    # Shutdown: persist the index
    logger.info("Shutting down — saving FAISS index...")
    vector_store.save()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Interstellar",
    description=(
        "Semantic & Change-Aware Satellite Imagery Search System. "
        "Search by meaning, detect changes, cluster similar sites, "
        "and review results — all fully offline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from app.routers import search, change, cluster, ingest, review, export

app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(change.router, prefix="/change", tags=["Change Detection"])
app.include_router(cluster.router, prefix="/cluster", tags=["Clustering"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(review.router, prefix="/review", tags=["Review"])
app.include_router(export.router, prefix="/export", tags=["Export"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Interstellar",
        "status": "online",
        "version": "1.0.0",
        "index_size": vector_store.get_index_size(),
        "offline_mode": True,
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
