import fnmatch
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


class MemoryRedis:
    """Small Redis-like fallback used when redis-py or Redis Server is unavailable."""

    def __init__(self):
        self._store: Dict[str, Tuple[str, Any, Optional[float]]] = {}

    def pipeline(self, transaction: bool = False):
        return _MemoryPipeline(self)

    def setex(self, key: str, ttl: int, value: Any):
        self._set(key, "string", value, ttl)
        return True

    def set(self, key: str, value: Any):
        self._set(key, "string", value, None)
        return True

    def get(self, key: str):
        item = self._get(key)
        if not item or item[0] != "string":
            return None
        return item[1]

    def mget(self, keys, *args):
        return [self.get(key) for key in _flatten_keys(keys, *args)]

    def delete(self, *keys):
        count = 0
        for key in _flatten_keys(*keys):
            if key in self._store:
                del self._store[key]
                count += 1
        return count

    def exists(self, key: str):
        return int(self._get(key) is not None)

    def expire(self, key: str, ttl: int):
        item = self._get(key)
        if not item:
            return False
        kind, value, _ = item
        self._store[key] = (kind, value, time.time() + ttl)
        return True

    def hset(self, name: str, mapping: Optional[Dict[str, Any]] = None, key=None, value=None):
        item = self._get(name)
        data = dict(item[1]) if item and item[0] == "hash" else {}
        if mapping:
            data.update(mapping)
        elif key is not None:
            data[key] = value
        self._store[name] = ("hash", data, item[2] if item else None)
        return len(data)

    def hgetall(self, key: str):
        item = self._get(key)
        if not item or item[0] != "hash":
            return {}
        return dict(item[1])

    def rpush(self, key: str, *values):
        item = self._get(key)
        data = list(item[1]) if item and item[0] == "list" else []
        data.extend(values)
        self._store[key] = ("list", data, item[2] if item else None)
        return len(data)

    def lpush(self, key: str, *values):
        item = self._get(key)
        data = list(item[1]) if item and item[0] == "list" else []
        for value in values:
            data.insert(0, value)
        self._store[key] = ("list", data, item[2] if item else None)
        return len(data)

    def lrange(self, key: str, start: int, stop: int):
        item = self._get(key)
        if not item or item[0] != "list":
            return []
        data = list(item[1])
        if stop == -1:
            return data[start:]
        return data[start : stop + 1]

    def keys(self, pattern: str):
        for key in list(self._store.keys()):
            self._purge_if_expired(key)
        return [key for key in self._store if fnmatch.fnmatch(key, pattern)]

    def flushdb(self):
        self._store.clear()

    def _set(self, key: str, kind: str, value: Any, ttl: Optional[int]):
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (kind, value, expires_at)

    def _get(self, key: str):
        self._purge_if_expired(key)
        return self._store.get(key)

    def _purge_if_expired(self, key: str):
        item = self._store.get(key)
        if item and item[2] is not None and item[2] <= time.time():
            del self._store[key]


class _MemoryPipeline:
    def __init__(self, redis_client: MemoryRedis):
        self.redis_client = redis_client
        self.commands: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self):
        results = []
        for name, args, kwargs in self.commands:
            results.append(getattr(self.redis_client, name)(*args, **kwargs))
        self.commands = []
        return results

    def __getattr__(self, name: str):
        def queue(*args, **kwargs):
            self.commands.append((name, args, kwargs))
            return self

        return queue


class SimpleVectorIndex:
    """Tiny persistent L2 vector index with stable external ids."""

    def __init__(self):
        self.ids: List[str] = []
        self.vectors: Dict[str, List[float]] = {}

    @classmethod
    def load(cls, path: str):
        index = cls()
        if not os.path.exists(path):
            return index
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError, TypeError):
            return index
        for item in payload.get("items", []):
            item_id = item.get("id")
            vector = item.get("vector")
            if item_id and isinstance(vector, list):
                index.upsert(item_id, vector)
        return index

    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": 1,
            "items": [{"id": item_id, "vector": self.vectors[item_id]} for item_id in self.ids],
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    def upsert(self, item_id: str, vector: Sequence[Number]):
        if item_id not in self.vectors:
            self.ids.append(item_id)
        self.vectors[item_id] = [float(value) for value in vector]

    def remove(self, item_id: str):
        if item_id in self.vectors:
            del self.vectors[item_id]
        self.ids = [existing_id for existing_id in self.ids if existing_id != item_id]

    def clear(self):
        self.ids = []
        self.vectors = {}

    def search(self, vector: Sequence[Number], k: int) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        query = [float(value) for value in vector]
        faiss_hits = _search_with_faiss(query, self.ids, self.vectors, k)
        if faiss_hits is not None:
            return faiss_hits
        scored = []
        for item_id in self.ids:
            scored.append((item_id, squared_l2(query, self.vectors[item_id])))
        scored.sort(key=lambda item: item[1])
        return scored[:k]


def make_redis_client(
    redis_url: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    redis_client: Any = None,
):
    if redis_client is not None:
        return redis_client
    try:
        import redis  # type: ignore

        if str(redis_url).startswith("redis://"):
            client = redis.Redis.from_url(redis_url, password=redis_password)
        else:
            client = redis.Redis(host=redis_url, port=redis_port, password=redis_password)
        client.ping()
        return client
    except Exception:
        return MemoryRedis()


def hash_text(value: Union[bytes, str]) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_key(namespace: str, *parts: Union[bytes, str]) -> str:
    joined = b":".join(part if isinstance(part, bytes) else part.encode("utf-8") for part in parts)
    return "%s:%s" % (namespace, hashlib.sha256(joined).hexdigest())


def set_json(redis_client, key: str, payload: Dict[str, Any], ttl: Optional[int] = None):
    value = json.dumps(payload, ensure_ascii=False)
    if ttl:
        return redis_client.setex(key, ttl, value)
    return redis_client.set(key, value)


def get_json(redis_client, key: str) -> Optional[Dict[str, Any]]:
    raw = redis_client.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def delete_pattern(redis_client, pattern: str) -> int:
    keys_method = getattr(redis_client, "keys", None)
    if not keys_method:
        return 0
    keys = keys_method(pattern)
    if not keys:
        return 0
    return redis_client.delete(*keys)


def refresh_ttl(redis_client, key: str, ttl: Optional[int]):
    if ttl:
        redis_client.expire(key, ttl)


def normalize_texts(texts: Union[str, bytes, Iterable[Union[str, bytes]]]) -> List[Union[str, bytes]]:
    if isinstance(texts, (str, bytes)):
        return [texts]
    return list(texts)


def normalize_vectors(vectors: Any, expected_count: Optional[int] = None) -> List[List[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    if isinstance(vectors, tuple):
        vectors = list(vectors)
    if not isinstance(vectors, list):
        raise ValueError("Embedding output must be a vector or a list of vectors")
    if not vectors:
        return []

    if _is_number(vectors[0]):
        normalized = [[float(value) for value in vectors]]
    else:
        normalized = []
        for vector in vectors:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            normalized.append([float(value) for value in vector])

    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            "Expected %s embedding vectors, got %s" % (expected_count, len(normalized))
        )
    return normalized


def embed_texts(embedding_method, texts: Union[str, List[str]]) -> List[List[float]]:
    text_list = [texts] if isinstance(texts, str) else list(texts)
    method = embedding_method or default_embedding
    try:
        raw_vectors = method(text_list)
    except TypeError:
        raw_vectors = [method(text) for text in text_list]
    return normalize_vectors(raw_vectors, expected_count=len(text_list))


def default_embedding(texts: Union[str, List[str]], dims: int = 64) -> List[List[float]]:
    text_list = [texts] if isinstance(texts, str) else list(texts)
    vectors = []
    for text in text_list:
        vector = [0.0] * dims
        for char in str(text).lower():
            vector[ord(char) % dims] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        vectors.append(vector)
    return vectors


def squared_l2(left: Sequence[Number], right: Sequence[Number]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def current_timestamp() -> float:
    return time.time()


def strip_internal_fields(message: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in message.items() if not key.startswith("_")}


def _flatten_keys(keys=None, *args) -> List[str]:
    if args:
        values = [keys] + list(args)
    elif isinstance(keys, (list, tuple, set)):
        values = list(keys)
    elif keys is None:
        values = []
    else:
        values = [keys]
    return [key.decode("utf-8") if isinstance(key, bytes) else key for key in values]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _search_with_faiss(
    query: List[float],
    ids: List[str],
    vectors: Dict[str, List[float]],
    k: int,
) -> Optional[List[Tuple[str, float]]]:
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    if not ids:
        return []
    matrix = np.array([vectors[item_id] for item_id in ids], dtype="float32")
    if len(matrix.shape) != 2 or matrix.shape[1] == 0:
        return []
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    distances, indices = index.search(
        np.array([query], dtype="float32"),
        min(k, len(ids)),
    )
    hits = []
    for distance, index_position in zip(distances[0], indices[0]):
        if index_position < 0:
            continue
        hits.append((ids[int(index_position)], float(distance)))
    return hits
