from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from ._core import (
        cache_key,
        current_timestamp,
        delete_pattern,
        get_json,
        make_redis_client,
        normalize_texts,
        normalize_vectors,
        refresh_ttl,
        set_json,
    )
except ImportError:
    from _core import (  # type: ignore
        cache_key,
        current_timestamp,
        delete_pattern,
        get_json,
        make_redis_client,
        normalize_texts,
        normalize_vectors,
        refresh_ttl,
        set_json,
    )


class EmbeddingsCache:
    """Exact-match embedding cache inspired by RedisVL's EmbeddingsCache."""

    def __init__(
        self,
        name: str = "embedding_cache",
        ttl: Optional[int] = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_client: Any = None,
        model_name: str = "default",
    ):
        self.name = name
        self.ttl = ttl
        self.model_name = model_name
        self.redis = make_redis_client(
            redis_url=redis_url,
            redis_port=redis_port,
            redis_password=redis_password,
            redis_client=redis_client,
        )

    def set(
        self,
        content: Union[bytes, str],
        model_name: Optional[str],
        embedding: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        model = model_name or self.model_name
        vector = normalize_vectors(embedding, expected_count=1)[0]
        key = self._make_cache_key(content, model)
        entry = {
            "entry_id": key.rsplit(":", 1)[-1],
            "content": content.hex() if isinstance(content, bytes) else content,
            "content_type": "bytes" if isinstance(content, bytes) else "str",
            "model_name": model,
            "embedding": vector,
            "inserted_at": current_timestamp(),
        }
        if metadata is not None:
            entry["metadata"] = metadata
        set_json(self.redis, key, entry, ttl if ttl is not None else self.ttl)
        return key

    def get(self, content: Union[bytes, str], model_name: Optional[str] = None):
        key = self._make_cache_key(content, model_name or self.model_name)
        return self.get_by_key(key)

    def get_by_key(self, key: str):
        entry = get_json(self.redis, key)
        if entry is not None:
            refresh_ttl(self.redis, key, self.ttl)
        return entry

    def mset(self, items: List[Dict[str, Any]], ttl: Optional[int] = None) -> List[str]:
        keys = []
        for item in items:
            keys.append(
                self.set(
                    content=item["content"],
                    model_name=item.get("model_name", self.model_name),
                    embedding=item["embedding"],
                    metadata=item.get("metadata"),
                    ttl=ttl,
                )
            )
        return keys

    def mget(self, contents: Iterable[Union[bytes, str]], model_name: Optional[str] = None):
        return [self.get(content, model_name or self.model_name) for content in contents]

    def exists(self, content: Union[bytes, str], model_name: Optional[str] = None) -> bool:
        key = self._make_cache_key(content, model_name or self.model_name)
        return bool(self.redis.exists(key))

    def drop(self, content: Union[bytes, str], model_name: Optional[str] = None) -> None:
        key = self._make_cache_key(content, model_name or self.model_name)
        self.redis.delete(key)

    def store(
        self,
        text: Union[List[str], str],
        embedding: Any,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        texts = normalize_texts(text)
        vectors = normalize_vectors(embedding, expected_count=len(texts))
        keys = []
        for content, vector in zip(texts, vectors):
            keys.append(self.set(content, model_name or self.model_name, vector, metadata))
        return keys

    def call(self, text: Union[List[str], str], model_name: Optional[str] = None):
        texts = normalize_texts(text)
        hits = self.mget(texts, model_name or self.model_name)
        return [hit["embedding"] if hit else None for hit in hits]

    def delete(self, text: Union[List[str], str], model_name: Optional[str] = None) -> int:
        texts = normalize_texts(text)
        keys = [self._make_cache_key(content, model_name or self.model_name) for content in texts]
        return self.redis.delete(*keys)

    def clear(self) -> int:
        return delete_pattern(self.redis, "%s:embedding:*" % self.name)

    def _make_cache_key(self, content: Union[bytes, str], model_name: str) -> str:
        return cache_key("%s:embedding" % self.name, content, model_name)


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(name="embedding_cache", ttl=360)
    embed_cache.store(text="hello world", embedding=[0.1, 0.2, 0.3])
    print(embed_cache.call(text="hello world"))
    print(embed_cache.delete(text="hello world"))
