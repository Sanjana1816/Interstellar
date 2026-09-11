"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class TileResult(BaseModel):
    """A single tile in search/cluster results."""
    tile_id: str
    aoi_id: str
    date: str
    sensor: str
    score: float = 0.0
    cloud_pct: float = 0.0
    thumbnail_url: str | None = None
    bounds_wkt: str | None = None
    file_path: str | None = None


class Filters(BaseModel):
    """Optional filters for search queries."""
    aoi_id: str | None = None
    date_start: str | None = None  # YYYY-MM-DD
    date_end: str | None = None
    sensor: str | None = None
    max_cloud_pct: float | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TextSearchRequest(BaseModel):
    """POST /search/text"""
    query: str = Field(..., min_length=1, description="Natural-language search query")
    k: int = Field(default=20, ge=1, le=100)
    filters: Filters | None = None


class ImageSearchRequest(BaseModel):
    """POST /search/image — image is sent as multipart form data, this holds filters."""
    k: int = Field(default=20, ge=1, le=100)
    filters: Filters | None = None


class SearchResponse(BaseModel):
    """Response for both text and image search."""
    query: str | None = None
    total_results: int
    results: List[TileResult]


# ---------------------------------------------------------------------------
# Change Detection
# ---------------------------------------------------------------------------

class ChangeAnalysisRequest(BaseModel):
    """POST /change/analyze"""
    aoi_id: str
    date_start: str
    date_end: str
    change_types: List[str] | None = None  # filter by type


class ChangeCandidateResult(BaseModel):
    """A single change candidate."""
    candidate_id: str
    tile_before: TileResult
    tile_after: TileResult
    change_type: str  # construction, clearance, water_change, new_road
    confidence: float = Field(ge=0.0, le=1.0)
    earliest_observed_date: str | None = None
    status: str = "pending"
    diff_mask_url: str | None = None


class ChangeAnalysisResponse(BaseModel):
    """Response for change analysis."""
    aoi_id: str
    date_range: str
    total_candidates: int
    candidates: List[ChangeCandidateResult]


# ---------------------------------------------------------------------------
# Clustering / Discovery
# ---------------------------------------------------------------------------

class ClusterSimilarRequest(BaseModel):
    """POST /cluster/similar"""
    tile_id: str
    k: int = Field(default=20, ge=1, le=100)


class ClusterSimilarResponse(BaseModel):
    """Response for find-similar."""
    source_tile_id: str
    total_results: int
    similar_tiles: List[TileResult]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """POST /ingest/add — file is sent as multipart form data, this holds metadata."""
    aoi_id: str = "default"
    sensor: str = "sentinel-2"
    date: str | None = None  # YYYY-MM-DD, extracted from filename if not provided
    scene_id: str | None = None


class IngestResponse(BaseModel):
    """Response for ingestion."""
    status: str
    tiles_added: int
    tile_ids: List[str]
    index_size_after: int
    message: str


# ---------------------------------------------------------------------------
# Review / Audit
# ---------------------------------------------------------------------------

class ReviewDecisionRequest(BaseModel):
    """POST /review/decision"""
    candidate_id: str | None = None
    tile_id: str | None = None
    reviewer: str = "analyst"
    decision: str = Field(..., pattern="^(accept|reject)$")
    note: str | None = None


class ReviewDecisionResponse(BaseModel):
    """Response for a review decision."""
    decision_id: str
    status: str
    message: str


class ReviewQueueItem(BaseModel):
    """A single item in the review queue."""
    candidate_id: str
    change_type: str
    confidence: float
    earliest_observed_date: str | None = None
    status: str
    tile_before: TileResult
    tile_after: TileResult
    decisions: List[dict] = []


class ReviewQueueResponse(BaseModel):
    """Response for the review queue."""
    total_pending: int
    items: List[ReviewQueueItem]


class ReviewHistoryItem(BaseModel):
    """A single audit trail entry."""
    decision_id: str
    candidate_id: str | None = None
    tile_id: str | None = None
    reviewer: str
    decision: str
    timestamp: str
    note: str | None = None


class ReviewHistoryResponse(BaseModel):
    """Response for review history."""
    total_decisions: int
    decisions: List[ReviewHistoryItem]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportResponse(BaseModel):
    """Response for result export with full provenance."""
    result_id: str
    source_scene_id: str | None = None
    tile_id: str | None = None
    candidate_id: str | None = None
    processing_steps: List[str]
    confidence: float | None = None
    change_type: str | None = None
    date: str | None = None
    sensor: str | None = None
    bounds_wkt: str | None = None
    review_decisions: List[dict]
    exported_at: str


# ---------------------------------------------------------------------------
# Admin / Stats
# ---------------------------------------------------------------------------

class SystemStats(BaseModel):
    """System statistics for the admin panel."""
    total_tiles: int
    total_aois: int
    total_change_candidates: int
    total_review_decisions: int
    index_size: int  # number of vectors in FAISS
    date_range: str | None = None
    sensors: List[str]
    last_updated: str | None = None
    offline_mode: bool = True
