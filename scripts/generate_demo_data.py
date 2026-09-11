"""
Script to generate synthetic multi-band demo data for offline testing.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

def generate_synthetic_scene(output_path: str, date: str, scenario: str):
    """
    Generate a 5-band (B02, B03, B04, B08, SCL) synthetic image as a numpy array.
    Shape: (5, 1024, 1024)
    """
    print(f"Generating synthetic scene: {scenario} for {date}")
    
    # Base terrain (1024x1024)
    H, W = 1024, 1024
    
    # Initialize bands (B02, B03, B04, B08, SCL)
    bands = np.zeros((5, H, W), dtype=np.float32)
    
    # B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    # Give them some base values
    bands[0] = np.random.normal(0.05, 0.01, (H, W)) # Blue
    bands[1] = np.random.normal(0.08, 0.01, (H, W)) # Green
    bands[2] = np.random.normal(0.06, 0.01, (H, W)) # Red
    bands[3] = np.random.normal(0.25, 0.05, (H, W)) # NIR (high for veg)
    bands[4] = np.ones((H, W)) * 4  # SCL: 4 = vegetation
    
    # Add a river (high green/blue, low NIR)
    for i in range(H):
        width = 40 + int(np.sin(i/50.0) * 10)
        center = 300 + int(np.cos(i/100.0) * 50)
        bands[0, i, center-width:center+width] = 0.1
        bands[1, i, center-width:center+width] = 0.12
        bands[2, i, center-width:center+width] = 0.04
        bands[3, i, center-width:center+width] = 0.02 # low NIR for water
        bands[4, i, center-width:center+width] = 6    # SCL: 6 = water
        
    if scenario == "after_construction":
        # Add a bright construction site
        cy, cx, cr = 600, 700, 80
        bands[0, cy-cr:cy+cr, cx-cr:cx+cr] = 0.2
        bands[1, cy-cr:cy+cr, cx-cr:cx+cr] = 0.2
        bands[2, cy-cr:cy+cr, cx-cr:cx+cr] = 0.25
        bands[3, cy-cr:cy+cr, cx-cr:cx+cr] = 0.15
        bands[4, cy-cr:cy+cr, cx-cr:cx+cr] = 5 # SCL: bare soils
        
        # Add a road
        bands[0, cy:cy+20, cx:W] = 0.15
        bands[1, cy:cy+20, cx:W] = 0.15
        bands[2, cy:cy+20, cx:W] = 0.15
        bands[3, cy:cy+20, cx:W] = 0.15
        bands[4, cy:cy+20, cx:W] = 5
        
    if scenario == "after_clearance":
        # Reduce NIR in a patch (vegetation loss)
        cy, cx, cr = 200, 800, 100
        bands[3, cy-cr:cy+cr, cx-cr:cx+cr] *= 0.3
        bands[4, cy-cr:cy+cr, cx-cr:cx+cr] = 5
        
    # SCL cloud in top left corner
    bands[4, 0:100, 0:100] = 9 # High probability cloud
    bands[0, 0:100, 0:100] += 0.5
    bands[1, 0:100, 0:100] += 0.5
    
    # Save as .npy
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), bands)
    
    # Save metadata
    meta = {
        "bounds": [77.0, 28.5, 77.1, 28.6],
        "crs": "EPSG:4326"
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)
        
    print(f"Saved to {out_path}")

def main():
    raw_dir = settings.data_raw_dir / "demo_aoi"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    generate_synthetic_scene(
        raw_dir / "S2_demo_2024-01-01.npy", 
        "2024-01-01", 
        "base"
    )
    
    generate_synthetic_scene(
        raw_dir / "S2_demo_2024-02-01.npy", 
        "2024-02-01", 
        "after_construction"
    )
    
if __name__ == "__main__":
    main()
