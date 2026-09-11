"""
Search router — text and image-based semantic search.

POST /search/text  — text query → CLIP text encoder → FAISS → ranked tiles
POST /search/image — image upload → CLIP image encoder → FAISS → ranked tiles
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Depends
from PIL import Image

from app.models import (
    TextSearchRequest,
    SearchResponse,
    TileResult,
    Filters,
)
from app.database import get_session, Tile
from app.services import embedding, vector_store
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _apply_filters(tiles: list, filters: Optional[Filters]) -> list:
    """Apply optional metadata filters to tile results."""
    if filters is None:
        return tiles

    filtered = tiles
    if filters.aoi_id:
        filtered = [t for t in filtered if t.get("aoi_id") == filters.aoi_id]
    if filters.date_start:
        filtered = [t for t in filtered if t.get("date", "") >= filters.date_start]
    if filters.date_end:
        filtered = [t for t in filtered if t.get("date", "") <= filters.date_end]
    if filters.sensor:
        filtered = [t for t in filtered if t.get("sensor") == filters.sensor]
    if filters.max_cloud_pct is not None:
        filtered = [t for t in filtered if t.get("cloud_pct", 0) <= filters.max_cloud_pct]

    return filtered


def _tile_to_result(tile_row, score: float = 0.0) -> TileResult:
    """Convert a DB tile row to a TileResult."""
    thumb_path = tile_row.thumbnail_path or ""
    # Build a thumbnail URL relative to the static mount
    if thumb_path:
        # Extract relative path from thumbnails dir
        try:
            from pathlib import Path
            rel = Path(thumb_path).relative_to(settings.data_thumbnails_dir)
            thumb_url = f"/thumbnails/{rel.as_posix()}"
        except (ValueError, Exception):
            thumb_url = thumb_path
    else:
        thumb_url = None

    return TileResult(
        tile_id=tile_row.tile_id,
        aoi_id=tile_row.aoi_id,
        date=tile_row.date,
        sensor=tile_row.sensor,
        score=score,
        cloud_pct=tile_row.cloud_pct or 0.0,
        thumbnail_url=thumb_url,
        bounds_wkt=tile_row.bounds_wkt,
        file_path=tile_row.file_path,
    )


@router.post("/text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search tiles by natural-language text query.
    Encodes the query with CLIP text encoder, searches FAISS, applies filters.
    """
    logger.info(f"Text search: '{request.query}' (k={request.k})")

    # Encode query
    query_vector = embedding.encode_text(request.query)

    # Get candidate tile IDs for filtering
    filter_ids = None
    if request.filters:
        session = get_session()
        query = session.query(Tile)
        if request.filters.aoi_id:
            query = query.filter(Tile.aoi_id == request.filters.aoi_id)
        if request.filters.date_start:
            query = query.filter(Tile.date >= request.filters.date_start)
        if request.filters.date_end:
            query = query.filter(Tile.date <= request.filters.date_end)
        if request.filters.sensor:
            query = query.filter(Tile.sensor == request.filters.sensor)
        if request.filters.max_cloud_pct is not None:
            query = query.filter(Tile.cloud_pct <= request.filters.max_cloud_pct)
        filter_ids = [t.tile_id for t in query.all()]
        session.close()

    # FAISS search
    results = vector_store.search(query_vector, k=request.k, tile_ids_filter=filter_ids)

    # Fetch tile metadata from DB
    session = get_session()
    tile_results = []
    for tile_id, score in results:
        tile_row = session.query(Tile).filter(Tile.tile_id == tile_id).first()
        if tile_row:
            tile_results.append(_tile_to_result(tile_row, score))
    session.close()

    return SearchResponse(
        query=request.query,
        total_results=len(tile_results),
        results=tile_results,
    )


@router.post("/image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    k: int = 20,
    aoi_id: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    sensor: Optional[str] = None,
):
    """
    Search tiles by image similarity.
    Encodes the uploaded image with CLIP image encoder, searches FAISS.
    """
    logger.info(f"Image search (k={k})")

    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Encode
    query_vector = embedding.encode_image(image)

    # Build filter list
    filter_ids = None
    if any([aoi_id, date_start, date_end, sensor]):
        session = get_session()
        query = session.query(Tile)
        if aoi_id:
            query = query.filter(Tile.aoi_id == aoi_id)
        if date_start:
            query = query.filter(Tile.date >= date_start)
        if date_end:
            query = query.filter(Tile.date <= date_end)
        if sensor:
            query = query.filter(Tile.sensor == sensor)
        filter_ids = [t.tile_id for t in query.all()]
        session.close()

    # FAISS search
    results = vector_store.search(query_vector, k=k, tile_ids_filter=filter_ids)

    # Fetch tile metadata
    session = get_session()
    tile_results = []
    for tile_id, score in results:
        tile_row = session.query(Tile).filter(Tile.tile_id == tile_id).first()
        if tile_row:
            tile_results.append(_tile_to_result(tile_row, score))
    session.close()

    return SearchResponse(
        query=f"Image: {file.filename}",
        total_results=len(tile_results),
        results=tile_results,
    )
