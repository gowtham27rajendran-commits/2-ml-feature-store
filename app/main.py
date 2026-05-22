from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from app.store.feature_store import (
    OnlineFeatureStore, FeatureRegistry, FeatureDefinition, FeatureValue
)
import time

app = FastAPI(title="ML Feature Store", version="1.0.0")

registry = FeatureRegistry()
online_store = OnlineFeatureStore()

# Seed some feature definitions
registry.register(FeatureDefinition("user_age_days", "int", -1, "Days since user signup"))
registry.register(FeatureDefinition("last_purchase_amount", "float", 3600, "Last purchase in USD, TTL=1h"))
registry.register(FeatureDefinition("click_rate_7d", "float", 86400, "Click-through rate, last 7 days"))


class IngestRequest(BaseModel):
    entity_id: str
    feature_name: str
    value: Any
    version: str = "1.0.0"


class BatchReadRequest(BaseModel):
    entity_ids: List[str]
    feature_names: List[str]


@app.post("/features/ingest")
def ingest(req: IngestRequest):
    fv = FeatureValue(
        feature_name=req.feature_name,
        entity_id=req.entity_id,
        value=req.value,
        timestamp=time.time(),
        version=req.version,
    )
    feat_def = registry.get(req.feature_name, req.version)
    ttl = feat_def.ttl_seconds if feat_def else -1
    online_store.write(fv, ttl_seconds=ttl)
    return {"status": "ok", "entity_id": req.entity_id, "feature": req.feature_name}


@app.get("/features/{entity_id}")
def get_all_features(entity_id: str):
    features = online_store.read_all(entity_id)
    if not features:
        raise HTTPException(status_code=404, detail="No features found for entity.")
    return {"entity_id": entity_id, "features": features}


@app.get("/features/{entity_id}/{feature_name}")
def get_feature(entity_id: str, feature_name: str, version: str = "1.0.0"):
    value = online_store.read(entity_id, feature_name, version)
    if value is None:
        raise HTTPException(status_code=404, detail="Feature not found.")
    return {"entity_id": entity_id, "feature_name": feature_name, "value": value}


@app.post("/features/batch")
def batch_read(req: BatchReadRequest):
    results = online_store.batch_read(req.entity_ids, req.feature_names)
    return {"results": results}


@app.get("/registry")
def list_registry():
    return {"features": registry.list_features()}


@app.get("/health")
def health():
    return {"status": "ok"}
