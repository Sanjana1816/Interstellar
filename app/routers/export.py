"""
Export router — export results with full provenance.

GET /export/{result_id} — Export a result bundle with source scene, processing steps, confidence.
"""

import logging
import datetime

from fastapi import APIRouter, HTTPException

from app.models import ExportResponse
from app.database import get_session, Tile, ChangeCandidate, ReviewDecision

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{result_id}", response_model=ExportResponse)
async def export_result(result_id: str):
    """
    Export a result with full provenance:
    - Source scene ID
    - Processing steps applied
    - Confidence score
    - Review decisions
    """
    session = get_session()

    # Try as a change candidate first
    candidate = (
        session.query(ChangeCandidate)
        .filter(ChangeCandidate.candidate_id == result_id)
        .first()
    )

    if candidate:
        # Get the after-tile for metadata
        tile = session.query(Tile).filter(Tile.tile_id == candidate.tile_after_id).first()

        # Get all review decisions for this candidate
        decisions = (
            session.query(ReviewDecision)
            .filter(ReviewDecision.candidate_id == result_id)
            .all()
        )

        decision_list = [
            {
                "decision_id": d.decision_id,
                "reviewer": d.reviewer,
                "decision": d.decision,
                "timestamp": d.timestamp.isoformat() if d.timestamp else "",
                "note": d.note,
            }
            for d in decisions
        ]

        processing_steps = []
        if tile and tile.processing_steps:
            import json
            try:
                processing_steps = json.loads(tile.processing_steps)
            except Exception:
                processing_steps = [tile.processing_steps]

        session.close()

        return ExportResponse(
            result_id=result_id,
            source_scene_id=tile.scene_id if tile else None,
            tile_id=candidate.tile_after_id,
            candidate_id=candidate.candidate_id,
            processing_steps=processing_steps,
            confidence=candidate.confidence,
            change_type=candidate.change_type,
            date=candidate.earliest_observed_date,
            sensor=tile.sensor if tile else None,
            bounds_wkt=tile.bounds_wkt if tile else None,
            review_decisions=decision_list,
            exported_at=datetime.datetime.utcnow().isoformat(),
        )

    # Try as a tile ID
    tile = session.query(Tile).filter(Tile.tile_id == result_id).first()

    if tile:
        decisions = (
            session.query(ReviewDecision)
            .filter(ReviewDecision.tile_id == result_id)
            .all()
        )

        decision_list = [
            {
                "decision_id": d.decision_id,
                "reviewer": d.reviewer,
                "decision": d.decision,
                "timestamp": d.timestamp.isoformat() if d.timestamp else "",
                "note": d.note,
            }
            for d in decisions
        ]

        processing_steps = []
        if tile.processing_steps:
            import json
            try:
                processing_steps = json.loads(tile.processing_steps)
            except Exception:
                processing_steps = [tile.processing_steps]

        session.close()

        return ExportResponse(
            result_id=result_id,
            source_scene_id=tile.scene_id,
            tile_id=tile.tile_id,
            candidate_id=None,
            processing_steps=processing_steps,
            confidence=None,
            change_type=None,
            date=tile.date,
            sensor=tile.sensor,
            bounds_wkt=tile.bounds_wkt,
            review_decisions=decision_list,
            exported_at=datetime.datetime.utcnow().isoformat(),
        )

    session.close()
    raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
