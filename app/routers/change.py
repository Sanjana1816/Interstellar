"""
Change detection router.

POST /change/analyze — Given AOI + date range, find tile pairs and detect changes.
"""

import logging
from typing import List

import numpy as np
from fastapi import APIRouter

from app.models import (
    ChangeAnalysisRequest,
    ChangeAnalysisResponse,
    ChangeCandidateResult,
    TileResult,
)
from app.database import get_session, Tile, ChangeCandidate
from app.services import change_detection, vector_store
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=ChangeAnalysisResponse)
async def analyze_changes(request: ChangeAnalysisRequest):
    """
    Run change detection on an AOI for a date range.

    Finds temporal tile pairs (same location, different dates),
    runs rule-based change detection, and returns ranked candidates.
    """
    logger.info(
        f"Change analysis: aoi={request.aoi_id}, "
        f"dates={request.date_start} to {request.date_end}"
    )

    session = get_session()

    # Find all tiles in the AOI and date range
    tiles = (
        session.query(Tile)
        .filter(
            Tile.aoi_id == request.aoi_id,
            Tile.date >= request.date_start,
            Tile.date <= request.date_end,
        )
        .order_by(Tile.date)
        .all()
    )

    if len(tiles) < 2:
        session.close()
        return ChangeAnalysisResponse(
            aoi_id=request.aoi_id,
            date_range=f"{request.date_start} to {request.date_end}",
            total_candidates=0,
            candidates=[],
        )

    # Group tiles by spatial position (using tile_id suffix as proxy)
    # Tiles with same index position across dates are pairs
    from collections import defaultdict

    tile_groups = defaultdict(list)
    for tile in tiles:
        # Extract position index from tile_id: aoi_date_sensor_XXXX → XXXX
        parts = tile.tile_id.rsplit("_", 1)
        pos_key = parts[-1] if len(parts) > 1 else tile.tile_id
        tile_groups[pos_key].append(tile)

    # Build temporal pairs: compare each consecutive date pair
    tile_pairs = []
    for pos_key, group in tile_groups.items():
        group.sort(key=lambda t: t.date)
        for i in range(len(group) - 1):
            tile_before = group[i]
            tile_after = group[i + 1]

            # Load tile data
            before_data = _load_tile_data(tile_before.file_path)
            after_data = _load_tile_data(tile_after.file_path)

            if before_data is not None and after_data is not None:
                # Get embedding distance
                embed_dist = _compute_embedding_distance(
                    tile_before.tile_id, tile_after.tile_id
                )

                tile_pairs.append(
                    (
                        {
                            "tile_id": tile_before.tile_id,
                            "date": tile_before.date,
                            "tile_data": before_data,
                        },
                        {
                            "tile_id": tile_after.tile_id,
                            "date": tile_after.date,
                            "tile_data": after_data,
                        },
                    )
                )

    # Run change detection
    candidates = change_detection.analyze_change_for_aoi(tile_pairs)

    # Filter by requested change types
    if request.change_types:
        candidates = [
            c for c in candidates if c["change_type"] in request.change_types
        ]

    # Store candidates in DB and build response
    response_candidates = []
    for cand in candidates:
        # Save to DB
        db_candidate = ChangeCandidate(
            candidate_id=cand["candidate_id"],
            tile_before_id=cand["tile_before_id"],
            tile_after_id=cand["tile_after_id"],
            change_type=cand["change_type"],
            confidence=cand["confidence"],
            earliest_observed_date=cand["earliest_observed_date"],
            status="pending",
            metadata_json=cand.get("details", ""),
        )
        session.merge(db_candidate)

        # Build response
        tile_before = session.query(Tile).filter(Tile.tile_id == cand["tile_before_id"]).first()
        tile_after = session.query(Tile).filter(Tile.tile_id == cand["tile_after_id"]).first()

        if tile_before and tile_after:
            response_candidates.append(
                ChangeCandidateResult(
                    candidate_id=cand["candidate_id"],
                    tile_before=_tile_to_result(tile_before),
                    tile_after=_tile_to_result(tile_after),
                    change_type=cand["change_type"],
                    confidence=cand["confidence"],
                    earliest_observed_date=cand["earliest_observed_date"],
                    status="pending",
                )
            )

    session.commit()
    session.close()

    return ChangeAnalysisResponse(
        aoi_id=request.aoi_id,
        date_range=f"{request.date_start} to {request.date_end}",
        total_candidates=len(response_candidates),
        candidates=response_candidates,
    )


def _load_tile_data(file_path: str) -> np.ndarray | None:
    """Load tile data from disk."""
    from pathlib import Path

    path = Path(file_path)
    if path.suffix == ".npy" and path.exists():
        return np.load(str(path))

    # Try with .npy extension
    npy_path = path.with_suffix(".npy")
    if npy_path.exists():
        return np.load(str(npy_path))

    return None


def _compute_embedding_distance(tile_id_1: str, tile_id_2: str) -> float | None:
    """Compute cosine distance between two tile embeddings."""
    v1 = vector_store.get_vector(tile_id_1)
    v2 = vector_store.get_vector(tile_id_2)

    if v1 is None or v2 is None:
        return None

    # Cosine distance = 1 - cosine_similarity
    dot = float(np.dot(v1, v2))
    return 1.0 - dot


def _tile_to_result(tile_row) -> TileResult:
    """Convert a DB Tile row to TileResult."""
    from app.routers.search import _tile_to_result as search_tile_to_result
    return search_tile_to_result(tile_row, score=0.0)
