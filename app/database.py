"""
Database models and session management.
Uses SQLite for MVP — schema is migration-ready for PostgreSQL + PostGIS.
"""

import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Enum,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import sessionmaker, relationship, DeclarativeBase
from app.config import settings


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Tile(Base):
    """A single processed image tile with its metadata and embedding reference."""

    __tablename__ = "tiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tile_id = Column(String(255), unique=True, nullable=False, index=True)
    aoi_id = Column(String(128), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)          # YYYY-MM-DD
    sensor = Column(String(32), nullable=False, index=True)         # sentinel-2, etc.
    bounds_wkt = Column(Text, nullable=True)                        # WKT geometry
    cloud_pct = Column(Float, default=0.0)
    file_path = Column(Text, nullable=False)                        # path to tile COG
    thumbnail_path = Column(Text, nullable=True)                    # path to PNG thumb
    embedding_id = Column(Integer, nullable=True)                   # FAISS vector index
    scene_id = Column(String(255), nullable=True)                   # original source scene
    processing_steps = Column(Text, nullable=True)                  # JSON list of steps applied
    bands = Column(String(64), nullable=True)                       # e.g. "B02,B03,B04,B08"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    changes_before = relationship(
        "ChangeCandidate", foreign_keys="ChangeCandidate.tile_before_id", back_populates="tile_before"
    )
    changes_after = relationship(
        "ChangeCandidate", foreign_keys="ChangeCandidate.tile_after_id", back_populates="tile_after"
    )

    __table_args__ = (
        Index("idx_tile_aoi_date", "aoi_id", "date"),
    )


class ChangeCandidate(Base):
    """A detected change between two temporal tiles."""

    __tablename__ = "change_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(String(255), unique=True, nullable=False, index=True)
    tile_before_id = Column(String(255), ForeignKey("tiles.tile_id"), nullable=False)
    tile_after_id = Column(String(255), ForeignKey("tiles.tile_id"), nullable=False)
    change_type = Column(
        String(32), nullable=False
    )  # construction, clearance, water_change, new_road
    confidence = Column(Float, nullable=False)
    earliest_observed_date = Column(String(10), nullable=True)
    status = Column(String(16), default="pending")  # pending, accepted, rejected
    diff_mask_path = Column(Text, nullable=True)    # path to the pixel-level diff mask
    metadata_json = Column(Text, nullable=True)     # extra info as JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    tile_before = relationship("Tile", foreign_keys=[tile_before_id], back_populates="changes_before")
    tile_after = relationship("Tile", foreign_keys=[tile_after_id], back_populates="changes_after")
    decisions = relationship("ReviewDecision", back_populates="candidate")


class ReviewDecision(Base):
    """An analyst's accept/reject decision — forms the audit trail."""

    __tablename__ = "review_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(255), unique=True, nullable=False, index=True)
    candidate_id = Column(
        String(255), ForeignKey("change_candidates.candidate_id"), nullable=True
    )
    tile_id = Column(String(255), ForeignKey("tiles.tile_id"), nullable=True)
    reviewer = Column(String(128), nullable=False, default="analyst")
    decision = Column(String(16), nullable=False)  # accept / reject
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    note = Column(Text, nullable=True)

    # Relationships
    candidate = relationship("ChangeCandidate", back_populates="decisions")


# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

def get_engine():
    """Create a SQLAlchemy engine for SQLite."""
    db_url = f"sqlite:///{settings.db_path}"
    return create_engine(db_url, echo=False, connect_args={"check_same_thread": False})


def init_db():
    """Create all tables if they don't exist."""
    settings.ensure_directories()
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session():
    """Get a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()
