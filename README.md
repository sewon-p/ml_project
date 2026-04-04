# ProbeDensity — Traffic Density Estimation from Probe Vehicles

Predict how congested a road is using only smartphone sensor data from vehicles driving on it.

ProbeDensity is an end-to-end traffic density estimation system that turns GPS + accelerometer trajectories from probe vehicles into per-link density estimates. The project spans simulation-based data generation, feature engineering, model comparison, real-time serving, and a deployable multi-probe aggregation pipeline for real roads. Its focus is probe-based traffic density estimation under realistic road-network constraints, with the multi-probe method treated as one part of that larger system.

**[Live Demo](https://traffic-estimator-gcbqhrztha-du.a.run.app/)** · **[API Docs](https://traffic-estimator-gcbqhrztha-du.a.run.app/docs)** · **[Map](https://traffic-estimator-gcbqhrztha-du.a.run.app/map)** · **[ML Pipeline](https://traffic-estimator-gcbqhrztha-du.a.run.app/ml-pipeline/)**

<p align="center">
  <img src="docs/images/map_demo.gif" width="80%" alt="Probe-based traffic density estimation on the Seoul arterial network">
</p>

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Problem](#problem)
- [System Architecture](#system-architecture)
- [ML Pipeline Workbench](#ml-pipeline-workbench)
- [ML Approach](#ml-approach)
- [Backend and Data Engineering](#backend-and-data-engineering)
- [Lessons Learned](#lessons-learned)
- [Tech Stack](#tech-stack)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)

---

## What This Project Does

1. **Generates labeled traffic data** — 35K SUMO scenarios × 5 probes = 176K samples of 6-channel trajectories (VX, VY, AX, AY, speed, brake)
2. **Engineers 31 features** from car-following theory — speed statistics, acceleration patterns, braking behavior, lateral dynamics, time-series properties
3. **Trains and compares 6 model families** — XGBoost, LightGBM, LSTM, CNN-1D, GPR, FD baselines under the same pipeline
4. **Studies multi-probe aggregation in two settings** — aligned 1 km probe slices for the research case, and overlap-aware link-level fusion for the deployment case
5. **Builds a link-level fusion system for deployment** — when probes do not share the same traversal boundaries, the system predicts per probe first and then aggregates unequal traversals at the road-link level
6. **Wraps the offline workflow in a dashboard** — scenario generation, feature toggles, model selection, run history, scatter plots, and feature-importance inspection in one GUI
7. **Serves link-level predictions** — FastAPI, GIS link matching (2.2K Seoul arterial links), rolling link aggregation, PostgreSQL, Kafka/Pub-Sub, Leaflet map

Solo end-to-end project: simulation → ML → backend → deployment.

## Problem

Traffic density — vehicles per kilometer — is the fundamental measure of road congestion. But measuring it traditionally requires **loop detectors, cameras, or radar** embedded in the road, which are expensive and cover only major corridors.

Probe vehicles (taxis, ride-hails, smartphones) are everywhere, but a single probe only observes its own trajectory. The core challenge: **can you estimate how many vehicles surround a probe, using only its speed, acceleration, and braking patterns?**

Systematic experiments across 6 model families showed that, in the current single-probe 1 km setup, accuracy repeatedly lands around R²≈0.45 regardless of algorithm. This motivated the shift to multi-probe fusion, but the project ultimately split that problem into two different settings: an aligned research setting where 5 probes share the same 1 km slice, and a deployed road-network setting where probes rarely share identical start/end boundaries. The important result is not just "ensemble helps," but how much of the aligned 5-probe gain survives after introducing a deployable link-level fusion layer.

---

## System Architecture

```mermaid
graph TB
    subgraph Offline["Offline ML Pipeline"]
        A[SUMO Scenario Gen] --> B[FCD Trajectory Collection]
        B --> C[Edie Ground Truth]
        C --> D[31-Feature Engineering]
        D --> E[Model Training & Comparison]
        E --> F[Multi-Probe Penetration Study]
    end

    subgraph Phone["Smartphone Client"]
        G[GPS + Accelerometer 1Hz] --> H[30s Local Buffer]
        H --> I[POST /ingest bulk]
    end

    subgraph Server["Backend Server"]
        I --> J[Kalman Sensor Fusion]
        J --> K[GIS Link Match]
        K --> L[LinkBuffer 1km Accumulation]
        L --> M[31-Feature Extraction]
        M --> N[XGBoost Inference]
        N --> O[CF-Weighted Ensemble]
    end

    subgraph Output["Storage & Display"]
        O --> P[(PostgreSQL + TimescaleDB)]
        O --> Q[Leaflet Map Dashboard]
        O --> R[Kafka / Pub-Sub]
    end

    E -.->|trained model| N
    F -.->|fusion logic| O
    K -.->|auto speed_limit, lanes| L
```

### Real-Time Inference Sequence

```mermaid
sequenceDiagram
    participant Phone as Smartphone
    participant Server as Backend
    participant GIS as LinkMatcher
    participant LB as LinkBuffer
    participant ML as XGBoost
    participant Ens as Ensemble
    participant Map as Map/DB

    Phone->>Phone: Collect GPS+Accel (1Hz)
    Phone->>Phone: Buffer 30 samples
    Phone->>Server: POST /ingest (bulk)

    loop Each sample
        Server->>Server: Kalman fusion (server-side)
        Server->>GIS: match(lat, lon)
        GIS-->>Server: link_id, lanes, speed_limit, length
        Server->>LB: Accumulate FCD + distance
    end

    alt Distance >= 1km
        LB-->>Server: LinkTraversal
        Server->>ML: 31 features → density
        ML-->>Server: density, cf_score
        Server->>Ens: Register per link (15-min window)
        Ens-->>Server: weighted link aggregation
        Server->>Map: Store + WebSocket push
        Server-->>Phone: Prediction result
    else Accumulating
        Server-->>Phone: Distance status
    end
```

### Key Design Decisions

**Link-based inference (not time-based)**: The system accumulates FCD as the probe traverses consecutive road links, triggering prediction at **1km+ distance** — not after a fixed time window. In deployment, different probes rarely cut that 1 km window at the same place, so the important engineering step is not just accumulation but the **link-level fusion layer** that aggregates density through the road links those unequal traversals overlap.

**Thin client, centralized processing (with tradeoffs)**: The phone buffers about 30s of raw GPS+accelerometer data and uploads it in bulk, while the server handles Kalman fusion, GIS matching, feature extraction, inference, and ensemble logic. This keeps map logic and model updates in one place instead of duplicating them across devices, but it also makes the system more dependent on backend availability and network delivery.

**Two-stage multi-probe design**: The research version studies aligned multi-probe aggregation when several probes observe the same 1 km slice. The deployed version cannot assume that alignment, so it first predicts density for each traversal and then aggregates those unequal traversals at the road-link level inside a rolling window.

---

## ML Pipeline Workbench

The offline workflow is not just script-driven. I built an ML pipeline dashboard so experiment work is manageable from one place: generate scenarios, resume from saved assets, adjust scenario distributions, choose feature sets, pick model families, and inspect evaluation output after training.

The GUI matters because this project has many interacting choices that are painful to juggle by hand. A run may change scenario counts, probes per scenario, FD residual settings, handcrafted feature groups, window features, and training models all at once. The dashboard turns those into a reproducible workbench instead of a long sequence of shell commands and config edits.

It also acts as an analysis surface after training:

- **From Scratch / Resume / Scenario Config** tabs cover new runs, partial reruns, and distribution-level scenario control.
- **Feature selection controls** let experiments include or exclude the 31 handcrafted features and window features without changing code.
- **Model selection** supports direct comparison across XGBoost, LightGBM, CNN-1D, LSTM, and window models.
- **Run history and inline results** keep completed runs explorable inside the UI.
- **Evaluation views** show per-model metrics, actual-vs-predicted scatter plots, and feature-importance charts so failure modes are easier to inspect.

<p align="center">
  <img src="docs/images/ml-pipeline.png" width="48%" alt="ML pipeline dashboard with inline evaluation results, scatter plot, and feature importance">
  <img src="docs/images/ml-pipeline-scenario.png" width="48%" alt="ML pipeline scenario configuration dashboard with network, demand, and vehicle parameter controls">
</p>

On the hosted server the dashboard is intentionally view-only, but locally it is the main interface for running and analyzing the ML pipeline.

---

## ML Approach

### Feature Engineering

31 features from car-following theory and traffic flow dynamics, registered via `@register_feature` decorator and selected through YAML config:

| Category | Features | Rationale |
|----------|----------|-----------|
| Speed statistics | mean, std, cv, iqr, min, max, median, p10, p90 | FD relationship proxy |
| Acceleration | ax_mean, ax_std, ay_mean, ay_std, jerk_mean, jerk_std | Car-following interaction intensity |
| Braking | brake_count, brake_time_ratio, mean_brake_duration | Congestion indicator |
| Stops | stop_count, stop_time_ratio, mean_stop_duration, slow_duration_ratio | Queue detection |
| Lateral | vy_mean, vy_std, vy_min, vy_max, vy_variance, vy_energy | Lane-change proxy |
| Time-series | speed_autocorr_lag1, speed_fft_dominant_freq, sample_entropy | Flow regime classification |

### Results

**Aligned multi-probe study** (1km, XGBoost, ideal same-slice setting):

| N (probes) | R² | MAE (veh/km/lane) | vs baseline |
|------------|-----|-------------------|-------------|
| 1 | 0.457 | 2.57 | — |
| 2 | 0.531 | 2.20 | +16% |
| 3 | 0.604 | 2.00 | +32% |
| 5 | **0.671** | **1.80** | **+47%** |

MAE=1.80 means **1–2 vehicles per km per lane** error — approaching fixed loop-detector noise (±1–3 veh/km/lane). This table is the aligned research setting: probes are assumed to describe the same observation slice.

**Observation length** (N=5): 250m → R²=0.615, 500m → 0.647, 750m → 0.665, 1km → 0.671.

**Deployed link-level fusion** (N=5): simple mean 0.601, **CF-aware weighted fusion 0.622**.

The gap between **0.671** and **0.622** is the gap between two different problems:

- **0.671**: ideal same-slice fusion, where probes can be aligned to the same 1 km segment
- **0.622**: deployable fusion, where probes arrive on different link chains and must be combined after per-probe prediction

**Model comparison** (single probe): FD baseline <0, GPR 0.41, LSTM/CNN-1D/XGBoost clustered around **0.44-0.46** → production.

<p align="center">
  <img src="docs/images/probe_count_vs_r2.png" width="70%" alt="Probe count vs R²">
</p>

---

## Backend and Data Engineering

### Ingestion Pipeline

```mermaid
flowchart LR
    A["Smartphone<br/>1Hz GPS + accelerometer"] --> B["30s local buffer<br/>reduce network 30x"]
    B --> C["POST /ingest<br/>bulk upload"]
    C --> D["Kalman fusion<br/>GPS + accel -> [x, vx, y, vy]"]
    D --> E["GIS link match<br/>grid-indexed 2.2K MOCT links"]
    E --> F["LinkBuffer<br/>accumulate consecutive links"]
    F --> G["1 km reached<br/>31-feature extraction + XGBoost"]
    G --> H["Prediction registered on traversed links"]
    H --> I["PostgreSQL"]
    H --> J["WebSocket push"]
    H --> K["Kafka / Pub-Sub"]
    H --> L["15-min rolling link aggregation"]
```

### Optimization Decisions

| Optimization | What it does | Impact |
|-------------|-------------|--------|
| Grid spatial index | 0.001° cells, search 3×3 neighborhood only | O(2.2K) → O(9 cells), <1ms |
| Re-match skip | Don't re-query GIS until probe moves >30m | ~90% fewer GIS calls |
| 30s bulk ingest | Client buffers locally, sends batch | 30× fewer HTTP requests |
| Sticky link | Require confirmed link change before switching | Prevents GPS jitter traversals |
| Graceful degradation | DB/Kafka/GIS each optional | Prediction always available |

### Database Schema

```mermaid
erDiagram
    RoadLink {
        int id PK
        string link_id UK
        string road_rank
        float link_length_m
        int lanes
    }
    EnsembleResult {
        int id PK
        float ensemble_density
        int probe_count
        datetime window_start
        datetime window_end
        bool is_frozen
    }
    Prediction {
        int id PK
        float density
        float flow
        float cf_weight
        float traversal_time
    }
    FCDRecordRow {
        int id PK
        float time
        float speed
        float brake
    }
    RoadLink ||--o{ EnsembleResult : has
    RoadLink ||--o{ Prediction : has
    EnsembleResult ||--o{ Prediction : aggregates
    Prediction ||--o{ FCDRecordRow : contains
```

**Aggregation lifecycle**: new probe → find/create active link window → weighted update → extend window. No new probe within 15 min → freeze. Garbage-collected after 1 hour.

### Multi-Probe Aggregation

#### 1. Original aligned setting

In the research setting, multiple probes are aligned to the **same 1 km slice** before fusion. That means each probe prediction refers to the same observation target, so CF intensity can be used directly as the fusion weight. In that original setup, the weighting is applied at the same-slice aggregation stage and gives the aligned 5-probe result of **R² = 0.671**.

```math
\text{cf}_i = \sigma_{a_x} + r_{\text{brake}} + \text{CV}_{\text{speed}}
```

```math
w_i = \frac{\exp(\text{cf}_i)}{\sum_j \exp(\text{cf}_j)} \quad \text{(softmax)}
```

```math
\hat{k}_{\text{ensemble}} = \sum_i w_i \cdot \hat{k}_i
```

#### 2. Applied deployment setting

On real road links, probes do **not** share the same 1 km boundaries, so that same aligned fusion rule cannot be applied directly at the feature level. The deployed system therefore changes the order:

```math
\hat{k}_t = f_{\theta}(x_t)
```

```math
\hat{k}_{\mathrm{link}} = \sum_{t \in T(\mathrm{link})} \alpha_t \hat{k}_t,
\quad
\alpha_t = \frac{\exp(\mathrm{cf}_t)}{\sum_{j \in T(\mathrm{link})}\exp(\mathrm{cf}_j)}
```

First predict density for each traversal, then aggregate only the traversals whose windows overlap the same road link. In other words, the deployment version uses **post-hoc CF-aware weighted averaging** at the link level because perfectly aligned same-slice fusion is no longer available. Under that constraint, the deployed 5-probe system reaches **R² = 0.622** versus **0.601** for a simple mean.

### Sensor Fusion

2D Kalman filter per session: state `[x, vx, y, vy]` in equirectangular frame. GPS measurement update (σ=5m) + accelerometer control input (heading-rotated). Sessions garbage-collected after 10 min inactivity.

---

## Lessons Learned

- **The main contribution is two-stage because the research setting and the deployment setting are different.** The project first studies aligned multi-probe aggregation in the 5-probe, 1 km setting, then builds a link-level fusion system that preserves most of that gain once probes no longer share the same traversal boundaries.
- **In the current single-probe, 1 km setup, accuracy repeatedly lands around R²≈0.45.** Tested across XGBoost, LightGBM, LSTM, CNN-1D, GPR (4 kernels), window features, and density weighting, the present single-probe observation design kept converging near that range. This is a limitation of the current setup, not a universal ceiling for longer windows or richer sensing.
- **The current web demo leaves a lot of device-side compute unused.** In the browser-first version, the phone is mostly a thin client. If this moves into an installed app or an in-vehicle system, more of the buffering, sensor fusion, feature preparation, and filtering can run locally before upload, reducing server load and latency.
- **The deployed contribution is overlap-aware link fusion, not just a fixed 1 km window.** Real vehicles observe different cut points across the same road timeline, so the system first predicts each traversal, then aggregates density on the links those windows overlap instead of requiring perfectly matched segments.
- **The implementation problem was fusion of unequal traversal windows.** Real probes do not begin and end their useful 1 km observation at the same place, so the deployable algorithm had to predict first and fuse second instead of directly averaging aligned samples.
- **Direct spacing from ADAS or connected vehicles is the most promising next sensor upgrade.** In the aligned multi-probe study, **XGBoost (31 features + CF)** rises from **0.641 → 0.752 → 0.801 → 0.848** as the spacing structure becomes more visible. The main takeaway is not just "more probes," but that the current phone-only system is still inferring inter-vehicle gap indirectly from trajectory shape. If headway or forward-gap measurements were available from ADAS or connected-vehicle signals and fused with the current features, density estimation should improve much more sharply than with trajectory-only inputs alone.
- **Simulation produces almost no congestion without bottlenecks.** Only 48 of 176K samples showed v_ratio < 0.4. The single straight-link SUMO setup cannot generate realistic stop-and-go waves. Future work requires multi-link networks with lane drops, signals, and merge sections.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML** | XGBoost, LightGBM, PyTorch (CNN-1D, LSTM, DeepSets), GPyTorch, scikit-learn, SHAP |
| **Backend** | FastAPI, uvicorn, WebSocket, Pydantic, SQLAlchemy async |
| **Database** | PostgreSQL + TimescaleDB, asyncpg |
| **Streaming** | Apache Kafka, Google Cloud Pub/Sub |
| **Spatial** | MOCT standard links, grid-indexed matcher, GeoJSON, Leaflet.js |
| **Infra** | Docker, Cloud Run, Artifact Registry, Secret Manager, GitHub Actions |
| **Data** | Apache Parquet, NumPy NPZ, SUMO (TraCI), Edie's definitions |

> **Note:** The [live demo](https://traffic-estimator-gcbqhrztha-du.a.run.app/) runs in read-only mode — ML Pipeline execution is disabled on the hosted server. Clone and run locally to train models.

## Running the Project

### Local (recommended for development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_console.py
```

Then open:
- `http://localhost:8000/` — project overview
- `http://localhost:8000/map` — link density map
- `http://localhost:8000/mobile` — mobile probe collection
- `http://localhost:8000/ml-pipeline/` — ML training dashboard
- `http://localhost:8000/docs` — API schema

### Docker

```bash
docker-compose up -d
curl localhost:8000/health
```

### ML Pipeline (simulation → training → evaluation)

```bash
# Full pipeline
python scripts/run_all.py --config configs/default.yaml

# Or step by step
python scripts/generate_scenarios.py --config configs/simulation/scenarios.yaml
python scripts/run_simulation.py --config configs/simulation/scenarios.yaml  # requires SUMO
python scripts/extract_features.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

The ML Pipeline dashboard (`/ml-pipeline/`) provides a web UI for these steps with run versioning and resume support. On the hosted server, pipeline execution is disabled — clone and run locally.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | No | PostgreSQL async URL. Server runs without DB if unset |
| `CONFIG_PATH` | No | Model and GIS config path (default: `configs/default.yaml`) |
| `MIN_TRAVERSAL_DISTANCE_M` | No | Min link accumulation before prediction (default: 1000) |
| `KAFKA_BOOTSTRAP_SERVERS` | No | Kafka broker. Falls back to Pub/Sub or skips |

### CI/CD and Deployment

Pushes to `main` trigger CI:
1. **Lint** — `ruff check + format`
2. **Type check** — `mypy src/api/`
3. **Test** — `pytest` (145 tests × Python 3.11–3.13)

GitHub Release triggers CD:
1. **Build** — Docker image → GCP Artifact Registry
2. **Deploy** — Cloud Run (0–2 auto-scaling, 2 GiB memory)
3. **Verify** — health check on deployed URL

## Project Structure

```
src/
├── api/            FastAPI app, link-based ingest, ensemble, async DB
├── data/           Dataset loading, Parquet I/O, preprocessing
├── evaluation/     Metrics, SHAP, traffic state classification
├── features/       @register_feature registry, 7 feature modules
├── gis/            Grid-indexed MOCT link matcher (road hierarchy)
├── models/         XGBoost, LightGBM, CNN1D, LSTM, FD, multi-probe DeepSets
├── simulation/     SUMO network gen, FCD collection, Edie ground truth
├── streaming/      Kafka/Pub-Sub abstraction, Kalman sensor fusion
├── training/       TabularTrainer (GroupKFold), DLTrainer (PyTorch)
├── utils/          Config, logging, seed, checkpoints
└── visualization/  Plots, SHAP, model comparison

scripts/            Pipeline entry points (train, evaluate, extract, dashboard)
static/             Web pages (console, mobile, map, pipeline manager)
configs/            Hierarchical YAML (inheritable via _base_)
data/gis/           MOCT standard link GeoJSON (2.2K Seoul arterial links)
.github/workflows/  CI (lint+test+build) + CD (Cloud Run deploy)
```

## License

All rights reserved. This repository is shared for portfolio and evaluation purposes only. Not licensed for redistribution or reuse.
