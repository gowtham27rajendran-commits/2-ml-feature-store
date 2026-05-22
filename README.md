# Real-Time ML Feature Store

A production ML feature store that serves pre-computed features to models with <5ms latency, supporting both batch training and real-time inference workloads.

## Architecture

```
Data Sources (events, DB, streams)
        ↓
  Feature Pipelines (Python transforms)
        ↓
  ┌─────────────────────┐
  │   Feature Registry  │  ← tracks versions, schemas, lineage
  └─────────────────────┘
        ↓             ↓
  Online Store      Offline Store
  (Redis, <5ms)     (S3 + Parquet, batch training)
        ↓
  Model Inference API
```

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Online store | Redis hashes | O(1) lookup by entity_id, sub-ms latency |
| Offline store | Parquet on S3 | Columnar format, 10x faster than CSV for training |
| Feature versioning | Semantic versioning in registry | Reproducible training — model knows exact feature version used |
| Point-in-time joins | Explicit timestamp tracking | Prevents label leakage in training data |
| TTL on features | Per-feature configurable | Age-sensitive features (last_login=1h) vs stable ones (city=never) |

## Running Locally

```bash
docker-compose up -d
pip install -r requirements.txt
python -m app.main
```

## API

```
POST /features/ingest          — write feature values
GET  /features/{entity_id}     — get all features for an entity (online)
GET  /features/{entity_id}/{feature_name}  — get single feature
POST /features/batch           — bulk fetch for model batch inference
GET  /registry                 — list all registered features + schemas
```

## What to implement next

- [ ] Point-in-time correct feature retrieval for training (`app/store/pit_join.py`)
- [ ] Feature drift monitoring — alert when feature distribution shifts
- [ ] Backfill pipeline — compute historical features for new feature definitions
- [ ] gRPC interface for lower-latency model serving

## Interview Talking Points

**"What's the dual-store pattern and why?"**
Online store (Redis) serves real-time inference in <5ms. Offline store (S3/Parquet) serves batch training jobs. Without this split, you'd either have slow inference or impossibly large Redis memory usage.

**"How do you prevent training-serving skew?"**
The feature pipeline code is shared — same transform runs at write time (online) and at training time (offline). Skew happens when teams write separate transform logic for each path.

**"What is point-in-time correctness?"**
When training, you must only use features that were available BEFORE the label timestamp. If your label is "did user churn on day 30", your features must be the values from day 29, not day 31. Getting this wrong causes label leakage and models that can't generalize.
