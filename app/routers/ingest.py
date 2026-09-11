"""
Ingestion router — incremental addition of new imagery.

POST /ingest/add — Upload a GeoTIFF → preprocess → embed → add to index (no rebuild).
"""

import io
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form

from app.models import IngestResponse
from app.database import get_session, Tile, init_db
from app.services import preprocessing, embedding, vector_store
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/add", response_model=IngestResponse)
async def ingest_add(
    file: UploadFile = File(...),
    aoi_id: str = Form(default="default"),
    sensor: str = Form(default="sentinel-2"),
    date: str = Form(default=None),
    scene_id: str = Form(default=None),
):
    """
    Ingest new imagery incrementally:
    1. Save uploaded file
    2. Preprocess (cloud mask, tile, thumbnail)
    3. Generate embeddings
    4. Add to FAISS index (no full rebuild)
    5. Update database
    """
    logger.info(f"Ingesting: {file.filename}, aoi={aoi_id}, sensor={sensor}")

    # Save uploaded file to raw directory
    raw_dir = settings.data_raw_dir / aoi_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / file.filename

    contents = await file.read()
    with open(raw_path, "wb") as f:
        f.write(contents)

    logger.info(f"Saved {len(contents)} bytes to {raw_path}")

    # Preprocess
    tile_infos = preprocessing.preprocess_scene(
        geotiff_path=raw_path,
        aoi_id=aoi_id,
        sensor=sensor,
        date=date,
        scene_id=scene_id or file.filename,
    )

    if not tile_infos:
        return IngestResponse(
            status="warning",
            tiles_added=0,
            tile_ids=[],
            index_size_after=vector_store.get_index_size(),
            message="No valid tiles produced from the uploaded file.",
        )

    # Generate embeddings
    tile_data_list = [info["tile_data"] for info in tile_infos]
    embeddings = embedding.encode_images_batch(tile_data_list)

    # Add to FAISS index
    tile_ids = [info["tile_id"] for info in tile_infos]
    vector_store.add_embeddings(tile_ids, embeddings)

    # Save to database
    session = get_session()
    for i, info in enumerate(tile_infos):
        tile = Tile(
            tile_id=info["tile_id"],
            aoi_id=info["aoi_id"],
            date=info["date"],
            sensor=info["sensor"],
            bounds_wkt=info["bounds_wkt"],
            cloud_pct=info["cloud_pct"],
            file_path=info["file_path"],
            thumbnail_path=info["thumbnail_path"],
            embedding_id=i,
            scene_id=info["scene_id"],
            processing_steps=info["processing_steps"],
            bands=info["bands"],
        )
        session.merge(tile)

    session.commit()
    session.close()

    # Persist the updated index
    vector_store.save()

    logger.info(f"Ingested {len(tile_ids)} tiles. Index size: {vector_store.get_index_size()}")

    return IngestResponse(
        status="success",
        tiles_added=len(tile_ids),
        tile_ids=tile_ids,
        index_size_after=vector_store.get_index_size(),
        message=f"Successfully ingested {len(tile_ids)} tiles from {file.filename}",
    )


@router.post("/add_local", response_model=IngestResponse)
async def ingest_add_local(
    file_path: str,
    aoi_id: str = "default",
    sensor: str = "sentinel-2",
    date: str = None,
    scene_id: str = None,
):
    """
    Ingest imagery from a local file path (for batch processing / scripting).
    """
    path = Path(file_path)
    if not path.exists():
        return IngestResponse(
            status="error",
            tiles_added=0,
            tile_ids=[],
            index_size_after=vector_store.get_index_size(),
            message=f"File not found: {file_path}",
        )

    # Preprocess
    tile_infos = preprocessing.preprocess_scene(
        geotiff_path=path,
        aoi_id=aoi_id,
        sensor=sensor,
        date=date,
        scene_id=scene_id or path.stem,
    )

    if not tile_infos:
        return IngestResponse(
            status="warning",
            tiles_added=0,
            tile_ids=[],
            index_size_after=vector_store.get_index_size(),
            message="No valid tiles produced.",
        )

    # Generate embeddings
    tile_data_list = [info["tile_data"] for info in tile_infos]
    embeddings = embedding.encode_images_batch(tile_data_list)

    # Add to index
    tile_ids = [info["tile_id"] for info in tile_infos]
    vector_store.add_embeddings(tile_ids, embeddings)

    # Save to DB
    session = get_session()
    for i, info in enumerate(tile_infos):
        tile = Tile(
            tile_id=info["tile_id"],
            aoi_id=info["aoi_id"],
            date=info["date"],
            sensor=info["sensor"],
            bounds_wkt=info["bounds_wkt"],
            cloud_pct=info["cloud_pct"],
            file_path=info["file_path"],
            thumbnail_path=info["thumbnail_path"],
            embedding_id=i,
            scene_id=info["scene_id"],
            processing_steps=info["processing_steps"],
            bands=info["bands"],
        )
        session.merge(tile)
    session.commit()
    session.close()

    vector_store.save()

    return IngestResponse(
        status="success",
        tiles_added=len(tile_ids),
        tile_ids=tile_ids,
        index_size_after=vector_store.get_index_size(),
        message=f"Ingested {len(tile_ids)} tiles from {path.name}",
    )
