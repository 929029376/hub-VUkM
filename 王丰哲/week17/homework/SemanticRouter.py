import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from ._core import (
        SimpleVectorIndex,
        default_embedding,
        delete_pattern,
        embed_texts,
        get_json,
        hash_text,
        make_redis_client,
        refresh_ttl,
        set_json,
    )
except ImportError:
    from _core import (  # type: ignore
        SimpleVectorIndex,
        default_embedding,
        delete_pattern,
        embed_texts,
        get_json,
        hash_text,
        make_redis_client,
        refresh_ttl,
        set_json,
    )


@dataclass
class Route:
    name: str
    references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance_threshold: Optional[float] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Route name must not be empty")
        if not self.references or any(not reference.strip() for reference in self.references):
            raise ValueError("Route references must be non-empty strings")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "references": list(self.references),
            "metadata": dict(self.metadata),
            "distance_threshold": self.distance_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            name=data["name"],
            references=list(data.get("references", [])),
            metadata=dict(data.get("metadata", {})),
            distance_threshold=data.get("distance_threshold"),
        )


class SemanticRouter:
    """Semantic intent router with cached repeated query decisions."""

    def __init__(
        self,
        name: str = "semantic_router",
        routes: Optional[List[Route]] = None,
        embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
        ttl: Optional[int] = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        distance_threshold: float = 0.3,
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
        self.index_path = os.path.join(self.storage_dir, "%s_routes.index" % self.name)
        self.index = SimpleVectorIndex.load(self.index_path)
        self.routes: Dict[str, Route] = {}
        self.reference_to_route: Dict[str, str] = {}
        self._load_routes()
        if routes:
            for route in routes:
                self.add_route(route)

    def add_route(
        self,
        questions: Optional[Union[Route, List[str]]] = None,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None,
    ):
        if isinstance(questions, Route):
            route = questions
        else:
            if questions is None or target is None:
                raise ValueError("questions and target are required")
            route = Route(
                name=target,
                references=list(questions),
                metadata=metadata or {},
                distance_threshold=distance_threshold,
            )

        existing = self.routes.get(route.name)
        if existing:
            references = list(dict.fromkeys(existing.references + route.references))
            route = Route(
                name=route.name,
                references=references,
                metadata=route.metadata or existing.metadata,
                distance_threshold=route.distance_threshold
                if route.distance_threshold is not None
                else existing.distance_threshold,
            )
        self.routes[route.name] = route
        self._rebuild_index()
        self._save_routes()
        return route

    def route(self, question: str):
        if not question:
            return None

        cached = get_json(self.redis, self._query_cache_key(question))
        if cached:
            refresh_ttl(self.redis, self._query_cache_key(question), self.ttl)
            return cached.get("route")

        if not self.index.ids:
            return None

        vector = embed_texts(self.embedding_method, question)[0]
        for reference_id, distance in self.index.search(vector, max(len(self.index.ids), 1)):
            route_name = self.reference_to_route.get(reference_id)
            if not route_name:
                continue
            route = self.routes.get(route_name)
            if not route:
                continue
            threshold = (
                route.distance_threshold
                if route.distance_threshold is not None
                else self.distance_threshold
            )
            if distance <= threshold:
                set_json(
                    self.redis,
                    self._query_cache_key(question),
                    {"route": route.name, "distance": distance},
                    self.ttl,
                )
                return route.name
        return None

    def __call__(self, question: str):
        return self.route(question)

    def clear_routes(self):
        self.routes = {}
        self.reference_to_route = {}
        self.index.clear()
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)
        deleted = self.redis.delete(self._routes_key())
        deleted += delete_pattern(self.redis, "%s:router_cache:*" % self.name)
        return deleted

    def _rebuild_index(self):
        self.index.clear()
        self.reference_to_route = {}
        references = []
        reference_ids = []
        for route in self.routes.values():
            for reference in route.references:
                reference_id = self._reference_id(route.name, reference)
                references.append(reference)
                reference_ids.append(reference_id)
                self.reference_to_route[reference_id] = route.name

        if references:
            vectors = embed_texts(self.embedding_method, references)
            for reference_id, vector in zip(reference_ids, vectors):
                self.index.upsert(reference_id, vector)
        self.index.save(self.index_path)

    def _save_routes(self):
        payload = {"routes": [route.to_dict() for route in self.routes.values()]}
        set_json(self.redis, self._routes_key(), payload, self.ttl)

    def _load_routes(self):
        payload = get_json(self.redis, self._routes_key())
        if not payload:
            return
        self.routes = {
            route.name: route for route in [Route.from_dict(item) for item in payload.get("routes", [])]
        }
        self.reference_to_route = {}
        for route in self.routes.values():
            for reference in route.references:
                self.reference_to_route[self._reference_id(route.name, reference)] = route.name

    def _reference_id(self, route_name: str, reference: str) -> str:
        return hash_text("%s:%s:%s" % (self.name, route_name, reference))

    def _routes_key(self) -> str:
        return "%s:routes" % self.name

    def _query_cache_key(self, question: str) -> str:
        return "%s:router_cache:%s" % (self.name, hash_text(question))


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(questions=["Hi, good morning", "Hi, good afternoon"], target="greeting")
    router.add_route(questions=["How do I return an item?", "refund policy"], target="refund")
    print(router("Hi, good morning"))
