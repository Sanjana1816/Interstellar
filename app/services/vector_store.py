"""
FAISS-based vector store for semantic search.

Supports:
- Incremental add (no full rebuild needed)
- Cosine similarity search (via normalized inner product)
- Persistence to/from disk
- ID mapping between FAISS indices and tile IDs
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level index cache
_index = None
_id_map: dict = {}  # faiss_idx -> tile_id
_reverse_map: dict = {}  # tile_id -> faiss_idx
_next_id: int = 0


def _get_index_path() -> Path:
    return settings.index_dir / settings.faiss_index_file


def _get_id_map_path() -> Path:
    return settings.index_dir / settings.faiss_id_map_file


def initialize():
    """Initialize or load the FAISS index."""
    global _index, _id_map, _reverse_map, _next_id

    import faiss

    settings.ensure_directories()

    index_path = _get_index_path()
    id_map_path = _get_id_map_path()

    if index_path.exists() and id_map_path.exists():
        logger.info(f"Loading existing FAISS index from {index_path}")
        _index = faiss.read_index(str(index_path))

        with open(id_map_path, "r") as f:
            _id_map = json.load(f)

        # Convert string keys back to int
        _id_map = {int(k): v for k, v in _id_map.items()}
        _reverse_map = {v: int(k) for k, v in _id_map.items()}
        _next_id = max(_id_map.keys()) + 1 if _id_map else 0

        logger.info(f"Loaded index with {_index.ntotal} vectors, {len(_id_map)} IDs")
    else:
        logger.info(f"Creating new FAISS index (dim={settings.embedding_dim})")
        _index = faiss.IndexFlatIP(settings.embedding_dim)  # Inner product = cosine on normalized vectors
        _id_map = {}
        _reverse_map = {}
        _next_id = 0

    return _index


def add_embeddings(tile_ids: List[str], vectors: np.ndarray):
    """
    Add embeddings to the index incrementally.

    Args:
        tile_ids: List of tile ID strings
        vectors: Embedding matrix (N, embedding_dim), should be L2-normalized
    """
    global _next_id

    if _index is None:
        initialize()

    assert vectors.shape[0] == len(tile_ids), \
        f"Mismatch: {vectors.shape[0]} vectors vs {len(tile_ids)} tile_ids"
    assert vectors.shape[1] == settings.embedding_dim, \
        f"Wrong dim: expected {settings.embedding_dim}, got {vectors.shape[1]}"

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vectors = vectors / norms

    # Add to FAISS
    _index.add(vectors.astype(np.float32))

    # Update ID maps
    for i, tid in enumerate(tile_ids):
        idx = _next_id + i
        _id_map[idx] = tid
        _reverse_map[tid] = idx

    _next_id += len(tile_ids)

    logger.info(f"Added {len(tile_ids)} vectors. Index now has {_index.ntotal} vectors.")


def search(
    query_vector: np.ndarray,
    k: int = 20,
    tile_ids_filter: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Search for the k nearest neighbors.

    Args:
        query_vector: Normalized query embedding (embedding_dim,)
        k: Number of results to return
        tile_ids_filter: Optional list of tile IDs to restrict search to

    Returns:
        List of (tile_id, score) tuples, sorted by descending score
    """
    if _index is None:
        initialize()

    if _index.ntotal == 0:
        return []

    # Ensure 2D
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Normalize
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # Search more than k if we need to filter
    search_k = min(k * 3 if tile_ids_filter else k, _index.ntotal)

    distances, indices = _index.search(query_vector.astype(np.float32), search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        tid = _id_map.get(int(idx))
        if tid is None:
            continue
        if tile_ids_filter and tid not in tile_ids_filter:
            continue
        results.append((tid, float(dist)))
        if len(results) >= k:
            break

    return results


def get_vector(tile_id: str) -> Optional[np.ndarray]:
    """Get the stored embedding vector for a tile."""
    if _index is None:
        initialize()

    idx = _reverse_map.get(tile_id)
    if idx is None:
        return None

    import faiss
    # Reconstruct the vector
    try:
        vec = np.zeros((1, settings.embedding_dim), dtype=np.float32)
        _index.reconstruct(idx, vec[0])
        return vec[0]
    except Exception:
        return None


def get_index_size() -> int:
    """Return the number of vectors in the index."""
    if _index is None:
        initialize()
    return _index.ntotal


def save():
    """Persist the index and ID map to disk."""
    if _index is None:
        return

    import faiss

    settings.ensure_directories()
    index_path = _get_index_path()
    id_map_path = _get_id_map_path()

    faiss.write_index(_index, str(index_path))

    with open(id_map_path, "w") as f:
        json.dump({str(k): v for k, v in _id_map.items()}, f)

    logger.info(f"Saved index ({_index.ntotal} vectors) to {index_path}")


def reset():
    """Clear the index completely (for testing)."""
    global _index, _id_map, _reverse_map, _next_id

    import faiss

    _index = faiss.IndexFlatIP(settings.embedding_dim)
    _id_map = {}
    _reverse_map = {}
    _next_id = 0
    save()
    logger.info("Index reset.")
