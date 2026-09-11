# Interstellar — Satellite Imagery Intelligence

A fully offline system that lets analysts search satellite imagery **by meaning** (e.g. "newly built structures near a river") and automatically flags **meaningful change over time**, filtering out false alarms like clouds and shadows.

## Features

- **Semantic Search**: Text-to-image and image-to-image search powered by CLIP embeddings and FAISS vector index.
- **Change Detection**: Rule-based temporal change classification (construction, vegetation clearance, water extent shifts, new roads).
- **Analyst Workflow**: Review queue with before/after evidence, confidence scoring, and an immutable audit trail.
- **Clustering / Discovery**: "Find Similar" functionality using embedding-space kNN and HDBSCAN.
- **Incremental Ingestion**: Add new imagery to the index without a full rebuild.
- **Fully Offline**: Designed to run on-premises in airgapped environments.

## Getting Started

### 1. Prerequisites

You need Docker and Docker Compose installed. 

### 2. Run the System

```bash
# Start the API and UI services
docker-compose up --build -d
```

- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### 3. Generate Demo Data

Since downloading real Sentinel-2 data requires credentials, you can generate synthetic test data to verify the pipeline:

```bash
# Run inside the api container or locally
python scripts/generate_demo_data.py

# Build the initial index from the demo data
python scripts/build_index.py
```

## System Architecture

See `docs/architecture.md` for a detailed breakdown of the 10 components.

## Evaluation

See `docs/evaluation_report.md` for the performance metrics.
