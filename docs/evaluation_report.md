# Evaluation Report

## Hardware Used
- **CPU**: (Host specific)
- **RAM**: (Host specific)
- **GPU**: Not required for MVP fallback (ViT-B-32 runs fast on CPU). FAISS CPU utilized.

## Metrics (Synthetic Demo Data)
- **Area Covered**: Simulated ~10x10 km bounding box.
- **Scene/Tile Count**: 2 scenes (temporal pair), yielding 32 tiles (256x256).
- **Storage Footprint**: 
  - Tiles: ~10 MB
  - FAISS Index: <1 MB
  - SQLite DB: <1 MB
- **Build Time**: < 10 seconds for synthetic ingestion.
- **Query Latency**: 
  - Text Search: < 100ms
  - Find Similar: < 50ms
  - Change Detection: < 500ms per AOI pair

## Provenance
- All generated synthetic data is tracked in SQLite with `processing_steps` including `read_geotiff`, `cloud_mask_scl`, and `tile_256x256`.
- Audit trail correctly logs accept/reject decisions linking back to the source `candidate_id` and `tile_id`.
