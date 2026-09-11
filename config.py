"""
Central configuration for the Interstellar system.
All paths, model parameters, and system settings are managed here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


# Project root is one level up from this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings — override via environment variables or .env file."""

    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_tiles_dir: Path = PROJECT_ROOT / "data" / "tiles"
    data_thumbnails_dir: Path = PROJECT_ROOT / "data" / "thumbnails"
    data_basemap_dir: Path = PROJECT_ROOT / "data" / "basemap"
    models_dir: Path = PROJECT_ROOT / "models"
    index_dir: Path = PROJECT_ROOT / "index"
    db_path: Path = PROJECT_ROOT / "data" / "interstellar.db"

    # --- Tile Processing ---
    tile_size: int = 256
    tile_overlap: int = 0
    target_crs: str = "EPSG:4326"
    target_resolution: float = 10.0  # metres per pixel

    # --- Embedding Model ---
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    embedding_dim: int = 512
    # Set to a local .pt path to use RemoteCLIP instead of default OpenAI CLIP
    remoteclip_weights: str | None = None

    # --- FAISS ---
    faiss_index_file: str = "faiss.index"
    faiss_id_map_file: str = "id_map.json"
    faiss_use_gpu: bool = False

    # --- Change Detection Thresholds ---
    cd_ndvi_threshold: float = 0.25
    cd_ndwi_threshold: float = 0.20
    cd_band_diff_threshold: float = 0.15
    cd_min_change_area_pixels: int = 100
    cd_edge_density_threshold: float = 0.05
    cd_embedding_distance_weight: float = 0.3

    # --- Cloud Masking (SCL values to mask) ---
    scl_mask_values: List[int] = Field(default=[3, 8, 9, 10])
    # 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus

    # --- Supported Sensors ---
    supported_sensors: List[str] = Field(
        default=["sentinel-2", "sentinel-1", "landsat", "bhuvan"]
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # --- Streamlit ---
    streamlit_port: int = 8501

    # --- Search Defaults ---
    default_search_k: int = 20
    max_search_k: int = 100

    # --- Clustering ---
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3

    model_config = {"env_prefix": "INTERSTELLAR_", "env_file": ".env"}

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.data_raw_dir,
            self.data_tiles_dir,
            self.data_thumbnails_dir,
            self.data_basemap_dir,
            self.models_dir,
            self.index_dir,
            self.db_path.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
