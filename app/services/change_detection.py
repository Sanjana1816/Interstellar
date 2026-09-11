"""
Rule-based change detection module.

Detects meaningful changes between temporal tile pairs using:
- NDVI difference → vegetation clearance
- NDWI difference → water extent change
- Band difference + edge density → construction
- Linear feature detection → new roads

Includes false-alarm suppression via cloud masking, minimum area thresholds,
and embedding distance as an additional confidence signal.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings
from app.services.preprocessing import compute_ndvi, compute_ndwi

logger = logging.getLogger(__name__)


def analyze_change(
    tile_before_data: np.ndarray,
    tile_after_data: np.ndarray,
    bands_before: dict | None = None,
    bands_after: dict | None = None,
    embedding_distance: float | None = None,
) -> dict:
    """
    Detect and classify change between two temporal tiles.

    Args:
        tile_before_data: (C, H, W) array for the earlier date
        tile_after_data: (C, H, W) array for the later date
        bands_before: optional dict of named bands for the earlier date
        bands_after: optional dict of named bands for the later date
        embedding_distance: cosine distance between embeddings (0-2 scale)

    Returns:
        dict with keys: change_type, confidence, change_mask, details
    """
    # Build band dicts if not provided
    if bands_before is None:
        bands_before = _array_to_bands(tile_before_data)
    if bands_after is None:
        bands_after = _array_to_bands(tile_after_data)

    # Run all detectors
    detections = []

    # 1. NDVI-based clearance detection
    ndvi_result = _detect_ndvi_change(bands_before, bands_after)
    if ndvi_result:
        detections.append(ndvi_result)

    # 2. NDWI-based water change detection
    ndwi_result = _detect_ndwi_change(bands_before, bands_after)
    if ndwi_result:
        detections.append(ndwi_result)

    # 3. Band difference + edge density → construction
    construction_result = _detect_construction(tile_before_data, tile_after_data)
    if construction_result:
        detections.append(construction_result)

    # 4. Linear feature detection → new road
    road_result = _detect_new_road(tile_before_data, tile_after_data)
    if road_result:
        detections.append(road_result)

    if not detections:
        return {
            "change_type": "none",
            "confidence": 0.0,
            "change_mask": None,
            "details": "No significant change detected.",
        }

    # Pick the highest-confidence detection
    best = max(detections, key=lambda d: d["confidence"])

    # Boost or dampen confidence based on embedding distance
    if embedding_distance is not None:
        embed_factor = min(embedding_distance / 1.0, 1.0)  # normalize to 0-1
        embed_weight = settings.cd_embedding_distance_weight
        best["confidence"] = (
            best["confidence"] * (1 - embed_weight) + embed_factor * embed_weight
        )

    best["confidence"] = round(min(max(best["confidence"], 0.0), 1.0), 4)
    return best


def analyze_change_for_aoi(
    tile_pairs: List[Tuple[dict, dict]],
) -> List[dict]:
    """
    Run change analysis on multiple tile pairs for an AOI.

    Args:
        tile_pairs: List of (tile_before_info, tile_after_info) dicts
                    Each must have 'tile_data' key with numpy array

    Returns:
        List of change candidate dicts
    """
    candidates = []

    for before_info, after_info in tile_pairs:
        before_data = before_info.get("tile_data")
        after_data = after_info.get("tile_data")

        if before_data is None or after_data is None:
            continue

        result = analyze_change(before_data, after_data)

        if result["change_type"] != "none" and result["confidence"] > 0.1:
            candidate = {
                "candidate_id": f"chg_{uuid.uuid4().hex[:12]}",
                "tile_before_id": before_info.get("tile_id", "unknown"),
                "tile_after_id": after_info.get("tile_id", "unknown"),
                "change_type": result["change_type"],
                "confidence": result["confidence"],
                "earliest_observed_date": after_info.get("date", "unknown"),
                "details": result.get("details", ""),
            }
            candidates.append(candidate)

    # Sort by confidence descending
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _detect_ndvi_change(bands_before: dict, bands_after: dict) -> Optional[dict]:
    """Detect vegetation clearance via NDVI drop."""
    ndvi_before = compute_ndvi(bands_before)
    ndvi_after = compute_ndvi(bands_after)

    if ndvi_before is None or ndvi_after is None:
        return None

    ndvi_diff = ndvi_before - ndvi_after  # positive = vegetation loss

    # Mask NaN
    valid_mask = np.isfinite(ndvi_diff)
    ndvi_diff = np.where(valid_mask, ndvi_diff, 0.0)

    # Threshold
    change_mask = ndvi_diff > settings.cd_ndvi_threshold
    change_area = int(np.sum(change_mask))

    if change_area < settings.cd_min_change_area_pixels:
        return None

    # Confidence: based on mean NDVI drop in changed area and area size
    mean_drop = float(np.mean(ndvi_diff[change_mask]))
    area_fraction = change_area / (ndvi_diff.shape[0] * ndvi_diff.shape[1])
    confidence = min(mean_drop * 2, 1.0) * 0.7 + min(area_fraction * 10, 1.0) * 0.3

    return {
        "change_type": "clearance",
        "confidence": float(confidence),
        "change_mask": change_mask.astype(np.uint8),
        "details": f"Vegetation loss: mean NDVI drop={mean_drop:.3f}, area={change_area}px",
    }


def _detect_ndwi_change(bands_before: dict, bands_after: dict) -> Optional[dict]:
    """Detect water extent change via NDWI shift."""
    ndwi_before = compute_ndwi(bands_before)
    ndwi_after = compute_ndwi(bands_after)

    if ndwi_before is None or ndwi_after is None:
        return None

    ndwi_diff = np.abs(ndwi_after - ndwi_before)

    valid_mask = np.isfinite(ndwi_diff)
    ndwi_diff = np.where(valid_mask, ndwi_diff, 0.0)

    change_mask = ndwi_diff > settings.cd_ndwi_threshold
    change_area = int(np.sum(change_mask))

    if change_area < settings.cd_min_change_area_pixels:
        return None

    mean_shift = float(np.mean(ndwi_diff[change_mask]))
    area_fraction = change_area / (ndwi_diff.shape[0] * ndwi_diff.shape[1])
    confidence = min(mean_shift * 2.5, 1.0) * 0.6 + min(area_fraction * 10, 1.0) * 0.4

    return {
        "change_type": "water_change",
        "confidence": float(confidence),
        "change_mask": change_mask.astype(np.uint8),
        "details": f"Water extent change: mean NDWI shift={mean_shift:.3f}, area={change_area}px",
    }


def _detect_construction(
    data_before: np.ndarray, data_after: np.ndarray
) -> Optional[dict]:
    """Detect construction via band difference + edge density increase."""
    # Use mean across bands for overall brightness change
    if data_before.ndim == 3:
        mean_before = np.nanmean(data_before, axis=0)
        mean_after = np.nanmean(data_after, axis=0)
    else:
        mean_before = data_before.astype(np.float32)
        mean_after = data_after.astype(np.float32)

    # Normalize to 0-1 range
    max_val = max(np.nanmax(mean_before), np.nanmax(mean_after), 1.0)
    mean_before = mean_before / max_val
    mean_after = mean_after / max_val

    # Band difference
    band_diff = np.abs(mean_after - mean_before)
    band_diff = np.nan_to_num(band_diff, nan=0.0)

    # Edge density using simple gradient magnitude
    edge_before = _compute_edge_density(mean_before)
    edge_after = _compute_edge_density(mean_after)
    edge_increase = edge_after - edge_before

    # Combined detection: high band diff + edge increase
    combined = (band_diff > settings.cd_band_diff_threshold) & (
        edge_increase > settings.cd_edge_density_threshold
    )
    change_area = int(np.sum(combined))

    if change_area < settings.cd_min_change_area_pixels:
        return None

    mean_diff = float(np.mean(band_diff[combined]))
    mean_edge = float(np.mean(edge_increase[combined]))
    area_fraction = change_area / (combined.shape[0] * combined.shape[1])
    confidence = (
        min(mean_diff * 3, 1.0) * 0.4
        + min(mean_edge * 5, 1.0) * 0.3
        + min(area_fraction * 10, 1.0) * 0.3
    )

    return {
        "change_type": "construction",
        "confidence": float(confidence),
        "change_mask": combined.astype(np.uint8),
        "details": f"Construction: band_diff={mean_diff:.3f}, edge_increase={mean_edge:.3f}, area={change_area}px",
    }


def _detect_new_road(
    data_before: np.ndarray, data_after: np.ndarray
) -> Optional[dict]:
    """Detect new roads via linear feature detection (simplified Hough-like approach)."""
    if data_before.ndim == 3:
        gray_before = np.nanmean(data_before, axis=0)
        gray_after = np.nanmean(data_after, axis=0)
    else:
        gray_before = data_before.astype(np.float32)
        gray_after = data_after.astype(np.float32)

    # Normalize
    max_val = max(np.nanmax(gray_before), np.nanmax(gray_after), 1.0)
    gray_before = np.nan_to_num(gray_before / max_val, nan=0.0)
    gray_after = np.nan_to_num(gray_after / max_val, nan=0.0)

    # Difference
    diff = np.abs(gray_after - gray_before)

    # Linear feature: look for elongated bright regions in the difference
    # Simplified: check if changed pixels form elongated shapes
    change_mask = diff > settings.cd_band_diff_threshold
    change_area = int(np.sum(change_mask))

    if change_area < settings.cd_min_change_area_pixels:
        return None

    # Compute aspect ratio of changed region using bounding box
    rows, cols = np.where(change_mask)
    if len(rows) == 0:
        return None

    row_range = rows.max() - rows.min() + 1
    col_range = cols.max() - cols.min() + 1
    aspect_ratio = max(row_range, col_range) / (min(row_range, col_range) + 1)

    # Roads are elongated: aspect ratio > 3 and fill ratio is low
    fill_ratio = change_area / (row_range * col_range + 1)

    if aspect_ratio < 3.0 or fill_ratio > 0.5:
        return None

    confidence = (
        min(aspect_ratio / 10.0, 1.0) * 0.5
        + (1.0 - fill_ratio) * 0.3
        + min(change_area / 1000.0, 1.0) * 0.2
    )

    return {
        "change_type": "new_road",
        "confidence": float(confidence),
        "change_mask": change_mask.astype(np.uint8),
        "details": f"Linear feature: aspect_ratio={aspect_ratio:.1f}, fill={fill_ratio:.2f}, area={change_area}px",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _array_to_bands(data: np.ndarray) -> dict:
    """Convert a (C, H, W) array to a band dict."""
    if data.ndim == 2:
        return {"B04": data}

    band_names = ["B02", "B03", "B04", "B08", "SCL"]
    bands = {}
    for i in range(min(data.shape[0], len(band_names))):
        bands[band_names[i]] = data[i]
    return bands


def _compute_edge_density(img: np.ndarray) -> np.ndarray:
    """Compute edge density using Sobel-like gradient magnitude."""
    img = np.nan_to_num(img, nan=0.0)

    # Simple Sobel approximation
    gy = np.diff(img, axis=0, prepend=img[:1, :])
    gx = np.diff(img, axis=1, prepend=img[:, :1])

    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude
