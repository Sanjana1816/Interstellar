"""
Clustering and discovery service.

Provides:
- kNN "find similar" via FAISS
- HDBSCAN-based clustering over all embeddings
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

from app.config import settings
from app.services import vector_store

logger = logging.getLogger(__name__)


def find_similar(tile_id: str, k: int = 20) -> List[Tuple[str, float]]:
    """
    Find the k most similar tiles to a given tile.

    Args:
        tile_id: The source tile ID
        k: Number of similar tiles to return

    Returns:
        List of (tile_id, similarity_score) tuples, excluding the source tile
    """
    # Get the source tile's embedding
    query_vector = vector_store.get_vector(tile_id)

    if query_vector is None:
        logger.warning(f"No embedding found for tile {tile_id}")
        return []

    # Search for k+1 to account for the source tile being in results
    results = vector_store.search(query_vector, k=k + 1)

    # Filter out the source tile
    return [(tid, score) for tid, score in results if tid != tile_id][:k]


def cluster_all(
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> dict:
    """
    Cluster all indexed embeddings using HDBSCAN.

    Args:
        min_cluster_size: Minimum cluster size (default from settings)
        min_samples: Minimum samples (default from settings)

    Returns:
        dict mapping tile_id -> cluster_label (-1 = noise)
    """
    import hdbscan as hdb

    if min_cluster_size is None:
        min_cluster_size = settings.hdbscan_min_cluster_size
    if min_samples is None:
        min_samples = settings.hdbscan_min_samples

    # Collect all embeddings
    index_size = vector_store.get_index_size()
    if index_size == 0:
        return {}

    # Reconstruct all vectors from the index
    all_ids = []
    all_vectors = []

    for idx, tile_id in vector_store._id_map.items():
        vec = vector_store.get_vector(tile_id)
        if vec is not None:
            all_ids.append(tile_id)
            all_vectors.append(vec)

    if len(all_vectors) < min_cluster_size:
        logger.warning(
            f"Not enough vectors ({len(all_vectors)}) for clustering "
            f"(min_cluster_size={min_cluster_size})"
        )
        return {tid: 0 for tid in all_ids}

    vectors_matrix = np.vstack(all_vectors)

    # Run HDBSCAN
    clusterer = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors_matrix)

    result = {tid: int(label) for tid, label in zip(all_ids, labels)}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    logger.info(f"Clustering: {n_clusters} clusters, {n_noise} noise points")

    return result


def get_cluster_summary(cluster_labels: dict) -> dict:
    """
    Summarize clustering results.

    Returns:
        dict with cluster_id -> list of tile_ids
    """
    clusters = {}
    for tile_id, label in cluster_labels.items():
        label_key = str(label)
        if label_key not in clusters:
            clusters[label_key] = []
        clusters[label_key].append(tile_id)

    return {
        "total_clusters": len([k for k in clusters if k != "-1"]),
        "noise_tiles": len(clusters.get("-1", [])),
        "clusters": clusters,
    }
