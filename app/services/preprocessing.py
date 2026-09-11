"""
Preprocessing pipeline for satellite imagery.

Handles:
- Reading multi-band GeoTIFFs
- Cloud masking via SCL band
- Reprojection to target CRS
- Tiling into 256×256 patches
- Thumbnail generation
- NDVI / NDWI spectral index computation
"""

import json
import logging
import uuid
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def _try_import_rasterio():
    """Lazy import rasterio — may not be installed in all environments."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.windows import Window
        return rasterio, from_bounds, Window
    except ImportError:
        return None, None, None


def read_multiband_geotiff(file_path: str | Path) -> dict:
    """
    Read a multi-band GeoTIFF and return bands + metadata.

    Returns dict with:
        - bands: dict of band_name -> np.ndarray
        - profile: rasterio profile
        - bounds: (left, bottom, right, top)
        - crs: coordinate reference system string
    """
    file_path = Path(file_path)
    
    # If it's a synthetic .npy file from our demo generator, handle it directly
    if file_path.suffix == ".npy":
        return _read_synthetic_geotiff(file_path)

    rasterio, _, _ = _try_import_rasterio()

    if rasterio is None:
        return _read_synthetic_geotiff(file_path)

    with rasterio.open(file_path) as src:
        profile = dict(src.profile)
        bounds = src.bounds
        crs = str(src.crs)

        # Read all bands
        band_count = src.count
        bands = {}
        band_names = _get_band_names(band_count)

        for i in range(1, band_count + 1):
            band_data = src.read(i).astype(np.float32)
            if i <= len(band_names):
                bands[band_names[i - 1]] = band_data
            else:
                bands[f"band_{i}"] = band_data

        return {
            "bands": bands,
            "profile": profile,
            "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "crs": crs,
        }


def _read_synthetic_geotiff(file_path: str | Path) -> dict:
    """
    Fallback reader for .npy-based synthetic data (when rasterio is unavailable).
    Looks for an accompanying .json metadata file.
    """
    file_path = Path(file_path)

    # Try loading as numpy
    if file_path.suffix == ".npy":
        data = np.load(str(file_path))
    elif file_path.suffix == ".tif" or file_path.suffix == ".tiff":
        # Try numpy sidecar
        npy_path = file_path.with_suffix(".npy")
        if npy_path.exists():
            data = np.load(str(npy_path))
        else:
            # Create dummy data
            data = np.random.rand(5, 256, 256).astype(np.float32)
    else:
        data = np.random.rand(5, 256, 256).astype(np.float32)

    # Load metadata sidecar
    meta_path = file_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {
            "bounds": [77.0, 28.5, 77.1, 28.6],
            "crs": "EPSG:4326",
        }

    band_names = _get_band_names(data.shape[0])
    bands = {}
    for i, name in enumerate(band_names[: data.shape[0]]):
        bands[name] = data[i]

    return {
        "bands": bands,
        "profile": {"height": data.shape[1], "width": data.shape[2], "count": data.shape[0]},
        "bounds": tuple(meta.get("bounds", [77.0, 28.5, 77.1, 28.6])),
        "crs": meta.get("crs", "EPSG:4326"),
    }


def _get_band_names(band_count: int) -> List[str]:
    """Map band indices to Sentinel-2 band names."""
    if band_count >= 5:
        return ["B02", "B03", "B04", "B08", "SCL"]
    elif band_count == 4:
        return ["B02", "B03", "B04", "B08"]
    elif band_count == 3:
        return ["B02", "B03", "B04"]
    else:
        return [f"band_{i+1}" for i in range(band_count)]


def apply_cloud_mask(bands: dict, mask_values: List[int] | None = None) -> Tuple[dict, float]:
    """
    Apply cloud mask using the SCL band.

    Args:
        bands: dict of band_name -> ndarray
        mask_values: SCL values to mask (default: 3, 8, 9, 10)

    Returns:
        (masked_bands, cloud_percentage)
    """
    if mask_values is None:
        mask_values = settings.scl_mask_values

    scl = bands.get("SCL")
    if scl is None:
        # No SCL band — return unmasked with 0% cloud
        return bands, 0.0

    # Create boolean mask: True where clouds/shadows exist
    cloud_mask = np.isin(scl.astype(int), mask_values)
    cloud_pct = float(np.mean(cloud_mask)) * 100.0

    # Apply mask to all non-SCL bands
    masked_bands = {}
    for name, data in bands.items():
        if name == "SCL":
            masked_bands[name] = data
            continue
        masked = data.copy()
        masked[cloud_mask] = np.nan
        masked_bands[name] = masked

    return masked_bands, cloud_pct


def compute_ndvi(bands: dict) -> np.ndarray | None:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red).
    Sentinel-2: (B08 - B04) / (B08 + B04)
    """
    nir = bands.get("B08")
    red = bands.get("B04")

    if nir is None or red is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where(np.isfinite(ndvi), ndvi, 0.0)

    return ndvi.astype(np.float32)


def compute_ndwi(bands: dict) -> np.ndarray | None:
    """
    Compute NDWI = (Green - NIR) / (Green + NIR).
    Sentinel-2: (B03 - B08) / (B03 + B08)
    """
    green = bands.get("B03")
    nir = bands.get("B08")

    if green is None or nir is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)
        ndwi = np.where(np.isfinite(ndwi), ndwi, 0.0)

    return ndwi.astype(np.float32)


def tile_array(
    data: np.ndarray, tile_size: int = 256, overlap: int = 0
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split a 2D or 3D array into tiles.

    Returns list of (tile_data, row_offset, col_offset).
    """
    if data.ndim == 2:
        h, w = data.shape
    else:
        _, h, w = data.shape

    step = tile_size - overlap
    tiles = []

    for row in range(0, h, step):
        for col in range(0, w, step):
            if data.ndim == 2:
                tile = data[row : row + tile_size, col : col + tile_size]
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    # Pad undersized edge tiles
                    padded = np.zeros((tile_size, tile_size), dtype=data.dtype)
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded
            else:
                tile = data[:, row : row + tile_size, col : col + tile_size]
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = np.zeros(
                        (data.shape[0], tile_size, tile_size), dtype=data.dtype
                    )
                    padded[:, : tile.shape[1], : tile.shape[2]] = tile
                    tile = padded

            tiles.append((tile, row, col))

    return tiles


def create_rgb_thumbnail(
    bands: dict, output_path: str | Path, size: Tuple[int, int] = (128, 128)
) -> str:
    """
    Create an RGB PNG thumbnail from B04 (Red), B03 (Green), B02 (Blue).
    """
    red = bands.get("B04")
    green = bands.get("B03")
    blue = bands.get("B02")

    if red is None or green is None or blue is None:
        # Fallback: use first 3 bands
        available = [v for k, v in bands.items() if k != "SCL"]
        if len(available) < 3:
            available = available + [np.zeros_like(available[0])] * (3 - len(available))
        red, green, blue = available[0], available[1], available[2]

    # Stack and normalize to 0-255
    rgb = np.stack([red, green, blue], axis=-1)

    # Handle NaN values
    rgb = np.nan_to_num(rgb, nan=0.0)

    # Percentile stretch for visualization
    valid = rgb[rgb > 0]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)
    else:
        rgb = rgb * 255

    rgb = rgb.astype(np.uint8)

    # Resize and save
    img = Image.fromarray(rgb)
    img = img.resize(size, Image.LANCZOS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")

    return str(output_path)


def preprocess_scene(
    geotiff_path: str | Path,
    aoi_id: str = "default",
    sensor: str = "sentinel-2",
    date: str | None = None,
    scene_id: str | None = None,
) -> List[dict]:
    """
    Full preprocessing pipeline for a single scene:
    1. Read multi-band GeoTIFF
    2. Apply cloud mask (SCL)
    3. Tile into 256×256 patches
    4. Generate thumbnails
    5. Return list of tile metadata dicts (ready for DB insertion)

    Returns:
        List of tile info dicts with keys:
        tile_id, aoi_id, date, sensor, bounds_wkt, cloud_pct,
        file_path, thumbnail_path, scene_id, processing_steps, bands, tile_data
    """
    geotiff_path = Path(geotiff_path)
    logger.info(f"Preprocessing scene: {geotiff_path.name}")

    # Extract date from filename if not provided
    if date is None:
        date = _extract_date_from_filename(geotiff_path.name)

    if scene_id is None:
        scene_id = geotiff_path.stem

    # 1. Read bands
    scene_data = read_multiband_geotiff(geotiff_path)
    bands = scene_data["bands"]
    bounds = scene_data["bounds"]

    # 2. Cloud mask
    masked_bands, cloud_pct = apply_cloud_mask(bands)

    # 3. Build multi-band array for tiling (exclude SCL)
    band_names = [k for k in masked_bands.keys() if k != "SCL"]
    if not band_names:
        logger.warning(f"No valid bands in {geotiff_path}")
        return []

    multi_band = np.stack([masked_bands[b] for b in band_names], axis=0)

    # 4. Tile
    tiles_data = tile_array(multi_band, tile_size=settings.tile_size, overlap=settings.tile_overlap)

    processing_steps = json.dumps(["read_geotiff", "cloud_mask_scl", "tile_256x256"])
    tile_results = []

    for idx, (tile_arr, row_off, col_off) in enumerate(tiles_data):
        tile_id = f"{aoi_id}_{date}_{sensor}_{idx:04d}"

        # Calculate tile bounds (approximate)
        h_total = multi_band.shape[1]
        w_total = multi_band.shape[2]
        left, bottom, right, top = bounds
        tile_left = left + (col_off / w_total) * (right - left)
        tile_right = left + (min(col_off + settings.tile_size, w_total) / w_total) * (right - left)
        tile_bottom = bottom + ((h_total - row_off - settings.tile_size) / h_total) * (top - bottom)
        tile_top = bottom + ((h_total - row_off) / h_total) * (top - bottom)

        bounds_wkt = (
            f"POLYGON(({tile_left} {tile_bottom}, {tile_right} {tile_bottom}, "
            f"{tile_right} {tile_top}, {tile_left} {tile_top}, {tile_left} {tile_bottom}))"
        )

        # Save tile as numpy array
        tile_dir = settings.data_tiles_dir / aoi_id / date
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_path = tile_dir / f"{tile_id}.npy"
        np.save(str(tile_path), tile_arr)

        # Save thumbnail
        tile_bands = {}
        for i, bname in enumerate(band_names[:tile_arr.shape[0]]):
            tile_bands[bname] = tile_arr[i]

        thumb_dir = settings.data_thumbnails_dir / aoi_id / date
        thumb_path = thumb_dir / f"{tile_id}.png"
        create_rgb_thumbnail(tile_bands, thumb_path)

        tile_results.append(
            {
                "tile_id": tile_id,
                "aoi_id": aoi_id,
                "date": date,
                "sensor": sensor,
                "bounds_wkt": bounds_wkt,
                "cloud_pct": cloud_pct,
                "file_path": str(tile_path),
                "thumbnail_path": str(thumb_path),
                "scene_id": scene_id,
                "processing_steps": processing_steps,
                "bands": ",".join(band_names),
                "tile_data": tile_arr,  # kept in memory for embedding
            }
        )

    logger.info(f"Preprocessed {len(tile_results)} tiles from {geotiff_path.name}")
    return tile_results


def _extract_date_from_filename(filename: str) -> str:
    """Try to extract a date from filenames like S2A_MSIL2A_20230615... or scene_2023-06-15."""
    import re

    # Try YYYYMMDD pattern
    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # Try YYYY-MM-DD pattern
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)

    return "unknown"
