# Build Brief: Semantic & Change-Aware Satellite Imagery Search System

*Read this top to bottom before writing any code. It's meant to be handed to someone with no prior context and let them start building immediately.*

---

## 1. What You're Building (One Paragraph)

A system that lets an analyst search a satellite-imagery archive **by meaning** ("newly built structures near a river") instead of only by coordinates/date, and that automatically flags **meaningful change over time** (new construction, land clearing, water-level shifts, new roads) while filtering out fake changes caused by clouds, seasons, shadows, and sensor differences. It must run **entirely offline/on-premises**, support adding new imagery **without rebuilding the whole index**, and give analysts a **review queue** where they confirm/reject results — with everything traceable back to the original source scene.

---

## 2. Non-Negotiable Requirements

These come directly from the task spec. Nothing here can be dropped — only simplified in *how* it's implemented.

1. **Semantic & multimodal retrieval** — free-text search + image-to-image search over tiles, rank-ordered, filterable by area/date/sensor.
2. **Multi-temporal change analysis** — detect appearance/disappearance/expansion/contraction of features; classify change type (construction, clearance, water-extent, road); estimate earliest date the change is visible.
3. **False-alarm suppression** — treat season, lighting, cloud, haze, snow, shadow, radiometric drift, and misregistration as confounders, not real change. Favor precision over blind recall.
4. **Discovery & clustering** — from one example site, find other visually/semantically similar sites without a new manual query.
5. **Analyst workflow & provenance** — ranked review queue with before/after evidence, metadata, confidence, and history; accept/reject logged to an audit trail; exports retain source-scene provenance.
6. **Scale, incremental ingestion, sovereignty** — efficient vector index; add new imagery incrementally (no full rebuild); runs fully offline with network disabled during evaluation; ingests GeoTIFF/COG; only public/organiser-approved imagery.

Final deliverables: source code, an architecture note, the index-build/incremental-ingestion procedure, model & dataset provenance documentation, and a reproducible evaluation report (area covered, scene/tile count, build time, storage footprint, query latency, hardware used).

---

## 3. MVP Scope (Build This First — ~75%+ Coverage)

**Build in full:** text/image search, basic change detection with confidence + false-alarm filtering, clustering/"find similar," analyst review UI with accept/reject logging, incremental ingestion, fully offline runtime.

**Simplify:**
- Use **Sentinel-2 optical only** as the live demo backbone. Ingest one sample tile each from Sentinel-1, Landsat, and Bhuvan to prove multi-source support, but don't build full pipelines for all four.
- Classify change type with **rule-based heuristics** (NDVI drop → clearance, NDWI shift → water change, band-diff + edge density → construction) instead of a trained classifier.
- Use Sentinel-2's built-in **SCL band** for cloud/shadow masking instead of a custom quality model.
- Keep the evaluation report scoped to one AOI and one time window.

**Explicitly skip for now (mention as future work in the report):** SAR+optical fusion, a deep-learned change classifier, multi-region large-scale indexing.

---

## 4. Dataset (Mandatory)

All data must be public/organiser-approved — no classified, operational, or service-generated imagery.

| Source | Type | Access |
|---|---|---|
| Copernicus Sentinel-2 | Optical multispectral (10–20 m) | https://dataspace.copernicus.eu/ |
| Copernicus Sentinel-1 | SAR | https://dataspace.copernicus.eu/ or https://search.asf.alaska.edu/ |
| USGS Landsat Collection 2 | Optical multispectral | https://earthexplorer.usgs.gov/ |
| NRSC/ISRO Bhuvan | Open EO products | https://bhuvan.nrsc.gov.in/ |

**How to get pilot data (Sentinel-2, the primary source):**
1. Register at dataspace.copernicus.eu.
2. Use the Copernicus Browser (https://browser.dataspace.copernicus.eu/) to visually pick an AOI with a clearly visible change across 2–3 dates (construction site, reservoir, new road). Note the bounding box and dates.
3. Programmatically search/download via the CDSE OData API (`https://catalogue.dataspace.copernicus.eu/odata/v1/Products`) using an access token from `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`. Filter for `SENTINEL-2` L2A products, your AOI polygon, date range, and a cloud-cover ceiling (e.g. `<20%`).
4. Download the full product, extract the 10 m bands (B02/B03/B04/B08) and the SCL band.

Store everything under `data/raw/`, keep original filenames and metadata — provenance tracking starts here.

---

## 5. Tech Stack

| Layer | Choice |
|---|---|
| Core language | Python |
| Geospatial I/O | Rasterio, GDAL, Shapely, PyProj, rio-cogeo |
| ML framework | PyTorch |
| Embedding model | Pretrained remote-sensing multimodal model (e.g. RemoteCLIP or SkyCLIP) — open licence, weights downloaded once and packaged for offline use |
| Vector index | FAISS (simplest for MVP; Qdrant/Milvus if you want a server-based index with built-in incremental add) |
| Metadata store | PostgreSQL + PostGIS (or SQLite for MVP speed) |
| Backend/API | FastAPI |
| Frontend | Streamlit (fastest for MVP) or React + Leaflet/MapLibre if time allows |
| Packaging | Docker + docker-compose, tested with network fully disabled |

---

## 6. System Architecture (Pipeline)

```
Raw imagery (S2, S1, Landsat, Bhuvan)
        ↓
Preprocess & tile (cloud mask via SCL, reproject, 256x256 patches, save as COG)
        ↓
Generate embeddings (pretrained remote-sensing CLIP model, per tile per date)
        ↓
Vector index + metadata DB (FAISS + Postgres/SQLite, supports incremental add)
        ↓
   ┌────────────┬──────────────────┬───────────────────┐
   ↓            ↓                  ↓
Semantic     Change detection    Clustering /
search       (embedding distance  discovery
(text/image   + NDVI/NDWI rules,  (kNN / HDBSCAN
 query →      cloud-masked,       over embeddings)
 ranked hits) confidence scored)
   └────────────┴──────────────────┴───────────────────┘
        ↓
Analyst review UI (queue, before/after, accept/reject, audit log)
        ↓ (feedback)
        back into vector index / rerank
```

---

## 7. Backend / API — What to Expose

Build these as FastAPI endpoints (or direct function calls from Streamlit if skipping a separate API layer for MVP speed):

- `POST /search/text` — text query → embed with CLIP text encoder → FAISS nearest neighbors → apply AOI/date/sensor filters → return ranked tile list with thumbnails + metadata.
- `POST /search/image` — image query → same flow using the image encoder.
- `POST /change/analyze` — given AOI + date range → return list of change candidates: location, change type, confidence, earliest-observed date, before/after tile pairs.
- `POST /cluster/similar` — given a tile ID → return N nearest tiles by embedding distance (the "find similar" feature).
- `POST /ingest/add` — accept a new date's imagery for an AOI → run preprocessing + embedding → add to index incrementally (no rebuild). This is the endpoint you'll trigger live in the demo.
- `POST /review/decision` — log an analyst's accept/reject decision with timestamp, tile ID, and reviewer note → write to the audit trail table → optionally adjust rerank weights.
- `GET /export/{result_id}` — export a result bundle that includes the source scene ID, processing steps applied, and confidence — i.e., full provenance.

---

## 8. UI Requirements (Detailed)

Build four screens/tabs. Keep it functional over pretty — clarity wins demos.

### 8.1 Search Screen
- Text input box for natural-language queries (e.g., "large vehicle concentrations on open ground").
- Optional image upload for image-to-image search.
- Filter panel: AOI (draw on map or select preset), date range picker, sensor dropdown.
- Results: ranked grid or list of tile thumbnails, each showing score, date, sensor, and a "View on map" button.
- Clicking a result opens a detail panel with the full-resolution tile, metadata, and a **"Find similar"** button (triggers clustering).

### 8.2 Change Detection Screen
- AOI selector + date-range selector (reuse from Search screen).
- "Run change analysis" button.
- Results as a ranked list of change candidates, each showing:
  - Before/after image pair (side-by-side or slider/swipe view)
  - Change type label (construction / clearance / water / road)
  - Confidence score (numeric + visual bar)
  - Earliest observed date
  - "Send to review queue" button

### 8.3 Review Queue Screen (the analyst workflow core)
- A queue/table of pending candidates (from search flags or change detection), sorted by confidence or recency.
- Each row expands to show: before/after evidence, location (small map), acquisition time, sensor, confidence, processing history (what steps were applied — cloud mask, normalization, etc.).
- **Accept** / **Reject** buttons on each item — writes to the audit log with timestamp and (optionally) a free-text note.
- A visible "history" panel showing past decisions, so it's clear this feeds an audit trail.
- An "Export" button on accepted items that bundles the result with full provenance (source scene ID + processing steps).

### 8.4 Ingestion / Admin Screen
- Shows currently indexed AOIs, date coverage, and tile counts.
- A control to add new imagery (upload or point to a staged file) → triggers the incremental ingestion endpoint.
- Live index size, last-updated timestamp — useful to visibly demonstrate "index grew without a rebuild."
- A network status indicator (or just a note) confirming offline/on-prem operation.

Map component (all screens that show location): use Leaflet or MapLibre with a simple base layer cached locally (no live internet tile server, to respect the offline constraint) or a static image basemap for the AOI.

---

## 9. Data & Metadata Model (Minimum Fields)

Every tile record needs, at minimum:
- `tile_id`, `aoi_id`, `date`, `sensor` (S2/S1/Landsat/Bhuvan), `bounds` (geometry), `cloud_pct`, `file_path`, `embedding_id`

Every change-candidate record needs:
- `candidate_id`, `tile_before_id`, `tile_after_id`, `change_type`, `confidence`, `earliest_observed_date`

Every review-decision record needs:
- `decision_id`, `candidate_id` or `tile_id`, `reviewer`, `decision` (accept/reject), `timestamp`, `note`

---

## 10. Constraints You Must Respect

- **Offline operation**: after models and data are staged, the whole system must run with network access disabled. Test this explicitly — don't assume it'll work.
- **Licensing**: any pretrained model used must have its origin and licence documented, and weights must be packaged for offline use (downloaded once, stored in `models/`, never fetched at runtime).
- **Data provenance**: only public/organiser-approved imagery. No classified, operational, or service-generated data, ever.
- **Provenance in exports**: every exported result must be traceable to its original source scene and the processing steps applied to it.

---

## 11. Team & Roles (6 People)

| Person | Owns |
|---|---|
| A | Data ingestion + preprocessing (Stage 1–2) |
| B | Embedding model + vector index (Stage 3–4) |
| C | Change detection module |
| D | Search API + clustering/discovery |
| E | Frontend/UI (all four screens) |
| F | Infra, offline packaging, evaluation report, demo script |

---

## 12. Build Order (Suggested ~2-Week Timeline)

| Days | Focus |
|---|---|
| 1 | Pick AOI, download pilot data, set up repo/Docker/GPU |
| 2–3 | Ingestion + preprocessing pipeline (reproject, tile, mask, metadata DB) |
| 4–6 | Embeddings + vector index + working semantic search |
| 6–8 | Change detection + clustering modules |
| 8–10 | Analyst UI (all 4 screens) + feedback logging |
| 10–11 | Offline packaging + incremental ingestion demo |
| 11–12 | Evaluation report + provenance docs |
| 12–13 | Full offline end-to-end test, bug fixes |
| 13–14 | Demo rehearsal, polish, buffer |

---

## 13. Live Demo Script (Rehearse This)

1. Type a natural-language query → show ranked results with thumbnails and map locations.
2. Click a result → hit "Find similar" → show other matching sites appear instantly.
3. Switch to Change Detection → run it on the hero AOI → show the flagged change, its type, confidence, and earliest-observed date.
4. Open the Review Queue → accept one, reject one → point out the audit log entry.
5. Go to the Ingestion screen → add a new date's imagery live → show it searchable within seconds, without a full rebuild.
6. Disable the network (or show it was already off) → repeat a query to prove full on-prem operation.
7. End on the evaluation report screen/doc for a few seconds — area, scene count, latency, hardware.

---

## 14. Definition of Done

You're demo-ready when:
- [ ] A search query returns ranked, filterable results with thumbnails
- [ ] A change-detection run on the hero AOI returns a real, correctly-classified change with a confidence score and earliest date
- [ ] Clicking "find similar" on any result returns other matching sites
- [ ] The review queue shows before/after evidence, metadata, and accept/reject buttons that write to a visible audit log
- [ ] Adding new imagery updates the index without a full rebuild, and this is visibly demonstrable
- [ ] The entire system runs with the network disabled
- [ ] Samples from all four data sources exist in the index with correct provenance tags
- [ ] The evaluation report and architecture note are written and accurate
