"""
Clustering / discovery router.

POST /cluster/similar — Given a tile ID, find the N most similar tiles.
"""

import logging

from fastapi import APIRouter

from app.models import ClusterSimilarRequest, ClusterSimilarResponse, TileResult
from app.database import get_session, Tile
from app.services import clustering

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/similar", response_model=ClusterSimilarResponse)
async def find_similar(request: ClusterSimilarRequest):
    """
    Find tiles most similar to a given tile using embedding-space kNN.
    """
    logger.info(f"Find similar: tile_id={request.tile_id}, k={request.k}")

    results = clustering.find_similar(request.tile_id, k=request.k)

    # Fetch tile metadata
    session = get_session()
    tile_results = []
    for tile_id, score in results:
        tile_row = session.query(Tile).filter(Tile.tile_id == tile_id).first()
        if tile_row:
            tile_results.append(
                TileResult(
                    tile_id=tile_row.tile_id,
                    aoi_id=tile_row.aoi_id,
                    date=tile_row.date,
                    sensor=tile_row.sensor,
                    score=score,
                    cloud_pct=tile_row.cloud_pct or 0.0,
                    thumbnail_url=tile_row.thumbnail_path,
                    bounds_wkt=tile_row.bounds_wkt,
                )
            )
    session.close()

    return ClusterSimilarResponse(
        source_tile_id=request.tile_id,
        total_results=len(tile_results),
        similar_tiles=tile_results,
    )
