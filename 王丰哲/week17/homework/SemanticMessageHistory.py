import os
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from ._core import (
        SimpleVectorIndex,
        current_timestamp,
        default_embedding,
        embed_texts,
        get_json,
        hash_text,
        make_redis_client,
        refresh_ttl,
        set_json,
        strip_internal_fields,
    )
except ImportError:
    from _core import (  # type: ignore
        SimpleVectorIndex,
        current_timestamp,
        default_embedding,
        embed_texts,
        get_json,
        hash_text,
        make_redis_client,
        refresh_ttl,
        set_json,
        strip_internal_fields,
    )


class SemanticMessageHistory:
    """Session-scoped chat history with recent and semantic retrieval."""

    allowed_roles = {"system", "user", "llm", "assistant", "tool"}

    def __init__(
        self,
        name: str,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        ttl: Optional[int] = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        distance_threshold: float = 0.7,
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
        self.index_path = os.path.join(self.storage_dir, "%s_history.index" % self.name)
        self.index = SimpleVectorIndex.load(self.index_path)

    def get_history(self) -> List[Dict[str, Any]]:
        history = get_json(self.redis, self._history_key()) or {"messages": []}
        return [strip_internal_fields(message) for message in history.get("messages", [])]

    def add_message(self, message: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(message, list):
            return self.add_messages(message)
        return self.add_messages([message])

    def add_messages(self, messages: List[Dict[str, Any]]):
        if not messages:
            return []

        history_payload = get_json(self.redis, self._history_key()) or {"messages": []}
        existing_messages = history_payload.get("messages", [])
        vectors = embed_texts(self.embedding_method, [message.get("content", "") for message in messages])
        added_ids = []

        for message, vector in zip(messages, vectors):
            normalized = self._normalize_message(message)
            existing_messages.append(normalized)
            self.index.upsert(normalized["_id"], vector)
            added_ids.append(normalized["_id"])

        set_json(self.redis, self._history_key(), {"messages": existing_messages}, self.ttl)
        self.index.save(self.index_path)
        return added_ids

    def get_recent(
        self,
        role: Optional[Union[str, List[str]]] = None,
        top_k: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        messages = self.get_history()
        if role:
            roles = {role} if isinstance(role, str) else set(role)
            messages = [message for message in messages if message.get("role") in roles]
        if top_k:
            messages = messages[-top_k:]
        return messages

    def get_relevant(
        self,
        content: str,
        top_k: int = 10,
        role: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        if not content or not self.index.ids:
            return []

        message_by_id = {message["_id"]: message for message in self._raw_history()}
        roles = None
        if role:
            roles = {role} if isinstance(role, str) else set(role)

        vector = embed_texts(self.embedding_method, content)[0]
        hits = []
        for message_id, distance in self.index.search(vector, max(top_k * 3, top_k, 1)):
            message = message_by_id.get(message_id)
            if not message or distance > self.distance_threshold:
                continue
            if roles and message.get("role") not in roles:
                continue
            hit = strip_internal_fields(message)
            hit["distance"] = distance
            hits.append(hit)
            if len(hits) >= top_k:
                break

        refresh_ttl(self.redis, self._history_key(), self.ttl)
        return hits

    def delete_history(self, top_k: int = 10):
        kept = self._raw_history()[-top_k:]
        self.index.clear()
        if kept:
            vectors = embed_texts(self.embedding_method, [message["content"] for message in kept])
            for message, vector in zip(kept, vectors):
                self.index.upsert(message["_id"], vector)
        set_json(self.redis, self._history_key(), {"messages": kept}, self.ttl)
        self.index.save(self.index_path)
        return len(kept)

    def clear_history(self):
        self.index.clear()
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)
        return self.redis.delete(self._history_key())

    def _raw_history(self) -> List[Dict[str, Any]]:
        history = get_json(self.redis, self._history_key()) or {"messages": []}
        return history.get("messages", [])

    def _normalize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("message must be a dictionary")
        role = message.get("role")
        content = message.get("content")
        if role not in self.allowed_roles:
            raise ValueError("role must be one of %s" % sorted(self.allowed_roles))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        normalized = dict(message)
        normalized["_id"] = normalized.get("_id") or hash_text(
            "%s:%s:%s:%s" % (self.name, role, content, current_timestamp())
        )
        normalized["inserted_at"] = normalized.get("inserted_at", current_timestamp())
        return normalized

    def _history_key(self) -> str:
        return "semantic_history:%s" % self.name


if __name__ == "__main__":
    history = SemanticMessageHistory(name="my-session")
    history.clear_history()
    history.add_messages(
        [
            {"role": "user", "content": "hello, how are you?"},
            {"role": "llm", "content": "I'm doing fine, thanks."},
        ]
    )
    print(history.get_recent())
