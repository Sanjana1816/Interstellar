"""
Review / audit trail router.

POST /review/decision  — Log an accept/reject decision
GET  /review/queue     — Get pending review items
GET  /review/history   — Get audit trail
"""

import logging
import uuid
import datetime

from fastapi import APIRouter

from app.models import (
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewQueueResponse,
    ReviewQueueItem,
    ReviewHistoryResponse,
    ReviewHistoryItem,
    TileResult,
)
from app.database import get_session, ChangeCandidate, ReviewDecision, Tile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/decision", response_model=ReviewDecisionResponse)
async def submit_decision(request: ReviewDecisionRequest):
    """
    Log an analyst's accept/reject decision.
    Writes to the audit trail and updates the candidate status.
    """
    decision_id = f"dec_{uuid.uuid4().hex[:12]}"
    logger.info(
        f"Review decision: {request.decision} on candidate={request.candidate_id}, "
        f"tile={request.tile_id}, reviewer={request.reviewer}"
    )

    session = get_session()

    # Create the decision record
    decision = ReviewDecision(
        decision_id=decision_id,
        candidate_id=request.candidate_id,
        tile_id=request.tile_id,
        reviewer=request.reviewer,
        decision=request.decision,
        timestamp=datetime.datetime.utcnow(),
        note=request.note,
    )
    session.add(decision)

    # Update candidate status if applicable
    if request.candidate_id:
        candidate = (
            session.query(ChangeCandidate)
            .filter(ChangeCandidate.candidate_id == request.candidate_id)
            .first()
        )
        if candidate:
            candidate.status = "accepted" if request.decision == "accept" else "rejected"

    session.commit()
    session.close()

    return ReviewDecisionResponse(
        decision_id=decision_id,
        status="recorded",
        message=f"Decision '{request.decision}' recorded for candidate {request.candidate_id or request.tile_id}",
    )


@router.get("/queue", response_model=ReviewQueueResponse)
async def get_review_queue(sort_by: str = "confidence", limit: int = 50):
    """
    Get the pending review queue — candidates awaiting analyst decision.
    """
    session = get_session()

    query = session.query(ChangeCandidate).filter(
        ChangeCandidate.status == "pending"
    )

    if sort_by == "confidence":
        query = query.order_by(ChangeCandidate.confidence.desc())
    elif sort_by == "date":
        query = query.order_by(ChangeCandidate.earliest_observed_date.desc())
    else:
        query = query.order_by(ChangeCandidate.created_at.desc())

    candidates = query.limit(limit).all()

    items = []
    for cand in candidates:
        tile_before = session.query(Tile).filter(Tile.tile_id == cand.tile_before_id).first()
        tile_after = session.query(Tile).filter(Tile.tile_id == cand.tile_after_id).first()

        decisions = [
            {
                "decision_id": d.decision_id,
                "reviewer": d.reviewer,
                "decision": d.decision,
                "timestamp": d.timestamp.isoformat() if d.timestamp else "",
                "note": d.note,
            }
            for d in cand.decisions
        ]

        items.append(
            ReviewQueueItem(
                candidate_id=cand.candidate_id,
                change_type=cand.change_type,
                confidence=cand.confidence,
                earliest_observed_date=cand.earliest_observed_date,
                status=cand.status,
                tile_before=_tile_to_result(tile_before) if tile_before else TileResult(
                    tile_id=cand.tile_before_id, aoi_id="", date="", sensor=""
                ),
                tile_after=_tile_to_result(tile_after) if tile_after else TileResult(
                    tile_id=cand.tile_after_id, aoi_id="", date="", sensor=""
                ),
                decisions=decisions,
            )
        )

    session.close()

    return ReviewQueueResponse(
        total_pending=len(items),
        items=items,
    )


@router.get("/history", response_model=ReviewHistoryResponse)
async def get_review_history(limit: int = 100):
    """
    Get the audit trail — all past review decisions.
    """
    session = get_session()

    decisions = (
        session.query(ReviewDecision)
        .order_by(ReviewDecision.timestamp.desc())
        .limit(limit)
        .all()
    )

    items = [
        ReviewHistoryItem(
            decision_id=d.decision_id,
            candidate_id=d.candidate_id,
            tile_id=d.tile_id,
            reviewer=d.reviewer,
            decision=d.decision,
            timestamp=d.timestamp.isoformat() if d.timestamp else "",
            note=d.note,
        )
        for d in decisions
    ]

    session.close()

    return ReviewHistoryResponse(
        total_decisions=len(items),
        decisions=items,
    )


def _tile_to_result(tile_row) -> TileResult:
    """Convert a DB tile to TileResult."""
    return TileResult(
        tile_id=tile_row.tile_id,
        aoi_id=tile_row.aoi_id,
        date=tile_row.date,
        sensor=tile_row.sensor,
        score=0.0,
        cloud_pct=tile_row.cloud_pct or 0.0,
        thumbnail_url=tile_row.thumbnail_path,
        bounds_wkt=tile_row.bounds_wkt,
    )
