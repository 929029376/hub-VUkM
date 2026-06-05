import os
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from ._core import (
        SimpleVectorIndex,
        cache_key,
        current_timestamp,
        default_embedding,
        delete_pattern,
        embed_texts,
        get_json,
        make_redis_client,
        normalize_texts,
        refresh_ttl,
        set_json,
    )
except ImportError:
    from _core import (  # type: ignore
        SimpleVectorIndex,
        cache_key,
        current_timestamp,
        default_embedding,
        delete_pattern,
        embed_texts,
        get_json,
        make_redis_client,
        normalize_texts,
        refresh_ttl,
        set_json,
    )


class SemanticCache:
    """Semantic prompt-response cache with Redis metadata and vector search."""

    def __init__(
        self,
        name: str,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        ttl: Optional[int] = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        distance_threshold: float = 0.1,
        redis_client: Any = None,
        storage_dir: Optional[str] = None,
    ):
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method or default_embedding
        self.redis = make_redis_client(
            redis_url=redis_url,
            redis_port=redis_port,
            redis_password=redis_password,
            redis_client=redis_client,
        )
        self.storage_dir = storage_dir or "."
        self.index_path = os.path.join(self.storage_dir, "%s.index" % self.name)
        self.index = SimpleVectorIndex.load(self.index_path)

    def store(
        self,
        prompt: Union[str, List[str]],
        response: Union[str, List[str]],
        metadata: Optional[Union[Dict[str, Any], List[Optional[Dict[str, Any]]]]] = None,
    ) -> List[str]:
        prompts = [str(item) for item in normalize_texts(prompt)]
        responses = response if isinstance(response, list) else [response]
        if len(prompts) != len(responses):
            raise ValueError("prompt and response must have the same number of items")

        if isinstance(metadata, list):
            metadatas = metadata
        else:
            metadatas = [metadata] * len(prompts)

        vectors = embed_texts(self.embedding_method, prompts)
        keys = []
        for item_prompt, item_response, item_metadata, vector in zip(
            prompts, responses, metadatas, vectors
        ):
            key = self._entry_key(item_prompt)
            entry_id = key.rsplit(":", 1)[-1]
            payload = {
                "entry_id": entry_id,
                "prompt": item_prompt,
                "response": item_response,
                "inserted_at": current_timestamp(),
            }
            if item_metadata is not None:
                payload["metadata"] = item_metadata
            set_json(self.redis, key, payload, self.ttl)
            self.index.upsert(entry_id, vector)
            keys.append(key)
        self.index.save(self.index_path)
        return keys

    def check(self, prompt: str, top_k: int = 1):
        if not prompt or not self.index.ids:
            return []

        vector = embed_texts(self.embedding_method, prompt)[0]
        hits = []
        for entry_id, distance in self.index.search(vector, max(top_k, 1)):
            if distance > self.distance_threshold:
                continue
            key = self._key_from_entry_id(entry_id)
            payload = get_json(self.redis, key)
            if payload is None:
                continue
            refresh_ttl(self.redis, key, self.ttl)
            hit = dict(payload)
            hit["distance"] = distance
            hits.append(hit)
        return hits

    def call(self, prompt: str, top_k: int = 1):
        return self.check(prompt, top_k=top_k)

    def delete(self, prompt: Union[str, List[str]]) -> int:
        prompts = [str(item) for item in normalize_texts(prompt)]
        keys = [self._entry_key(item_prompt) for item_prompt in prompts]
        for key in keys:
            self.index.remove(key.rsplit(":", 1)[-1])
        self.index.save(self.index_path)
        return self.redis.delete(*keys)

    def clear_cache(self):
        deleted = delete_pattern(self.redis, "%s:semantic:*" % self.name)
        self.index.clear()
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)
        return deleted

    def clear(self):
        return self.clear_cache()

    def _entry_key(self, prompt: str) -> str:
        return cache_key("%s:semantic" % self.name, prompt)

    def _key_from_entry_id(self, entry_id: str) -> str:
        return "%s:semantic:%s" % (self.name, entry_id)


if __name__ == "__main__":
    cache = SemanticCache(name="semantic_cache", distance_threshold=0.2)
    cache.clear_cache()
    cache.store("hello world", "hello response")
    print(cache.check("hello world"))
