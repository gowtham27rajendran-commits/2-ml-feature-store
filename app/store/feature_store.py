import redis
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class FeatureDefinition:
    """
    Schema contract for a feature.
    Versioning: bump major on breaking schema change, minor on addition.
    Registry stores all historical versions — models record which version they trained on.
    """
    name: str
    dtype: str          # float, int, string, embedding
    ttl_seconds: int    # -1 = no expiry
    description: str
    version: str = "1.0.0"
    owner: str = "unknown"


@dataclass
class FeatureValue:
    feature_name: str
    entity_id: str      # user_id, item_id, session_id etc.
    value: Any
    timestamp: float    # unix epoch — critical for point-in-time joins
    version: str = "1.0.0"


class FeatureRegistry:
    def __init__(self):
        self._registry: Dict[str, FeatureDefinition] = {}

    def register(self, feature: FeatureDefinition):
        key = f"{feature.name}:{feature.version}"
        self._registry[key] = feature
        print(f"[Registry] Registered feature: {key}")

    def get(self, name: str, version: str = "1.0.0") -> Optional[FeatureDefinition]:
        return self._registry.get(f"{name}:{version}")

    def list_features(self) -> List[Dict]:
        return [asdict(f) for f in self._registry.values()]


class OnlineFeatureStore:
    """
    Redis-backed online store for real-time inference.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.client = redis.from_url(redis_url, decode_responses=True)

    def write(self, fv: FeatureValue, ttl_seconds: int = -1):
        key = f"feature:{fv.entity_id}"
        field = f"{fv.feature_name}:{fv.version}"
        payload = json.dumps({"value": fv.value, "ts": fv.timestamp})
        self.client.hset(key, field, payload)
        if ttl_seconds > 0:
            # TTL on the whole hash — per-field TTL not supported in Redis
            # Workaround: store expiry in the JSON payload, check at read time
            self.client.expire(key, ttl_seconds)

    def read(self, entity_id: str, feature_name: str, version: str = "1.0.0") -> Optional[Any]:
        key = f"feature:{entity_id}"
        field = f"{feature_name}:{version}"
        raw = self.client.hget(key, field)
        if not raw:
            return None
        data = json.loads(raw)
        return data["value"]

    def read_all(self, entity_id: str) -> Dict[str, Any]:
        """Fetch all features for an entity in a single Redis call."""
        key = f"feature:{entity_id}"
        raw = self.client.hgetall(key)
        result = {}
        for field, payload in raw.items():
            feature_name = field.split(":")[0]
            result[feature_name] = json.loads(payload)["value"]
        return result

    def batch_read(self, entity_ids: List[str], feature_names: List[str]) -> Dict[str, Dict]:
        """
        Pipeline multiple HGETALL calls.
        Redis pipeline sends all commands in one TCP round trip.
        For 1000 entities: 1 pipeline vs 1000 sequential calls.
        """
        pipe = self.client.pipeline()
        for entity_id in entity_ids:
            pipe.hgetall(f"feature:{entity_id}")
        results = pipe.execute()
        return {
            entity_id: {
                f.split(":")[0]: json.loads(v)["value"]
                for f, v in raw.items()
                if f.split(":")[0] in feature_names
            }
            for entity_id, raw in zip(entity_ids, results)
        }


class OfflineFeatureStore:
    """
    S3 + Parquet offline store for batch training.
    """
    def write_batch(self, features):
        # TODO: use pandas + pyarrow to write partitioned Parquet
        # Partition by: date → entity_type → feature_name
        # This layout optimises for time-range scans in training jobs
        raise NotImplementedError("Implement: write to S3 as Parquet")

    def point_in_time_join(self, entity_df, feature_names, timestamp_col):
        """
        For each row in entity_df, fetch feature values that were available
        AT or BEFORE the row's timestamp. This prevents label leakage.

        Naive approach: for each entity, binary search feature history by timestamp.
        Scalable approach: SQL window function (LAST_VALUE ... OVER (... ORDER BY ts))
        """
        raise NotImplementedError("Implement: point-in-time correct feature join")
