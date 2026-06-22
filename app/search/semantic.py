import hashlib
from threading import Lock

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app import config

_model = None
_model_lock = Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def _fake_vector(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(config.VECTOR_SIZE).astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


def encode(texts: list[str]) -> np.ndarray:
    if config.FAKE_EMBED:
        return np.vstack([_fake_vector(t) for t in texts])
    emb = _get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def _make_client() -> QdrantClient:
    if config.QDRANT_URL == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=config.QDRANT_URL)


_client = _make_client()


def _collection(tenant: str) -> str:
    return f"catalog__{tenant}"


def _point_id(product_id: str) -> int:
    # Qdrant требует int/UUID; берём стабильный хэш строкового id.
    return int(hashlib.sha256(product_id.encode("utf-8")).hexdigest()[:15], 16)


def ensure_collection(tenant: str):
    name = _collection(tenant)
    if not _client.collection_exists(name):
        _client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE),
        )


def index(tenant: str, products):
    ensure_collection(tenant)
    vectors = encode([f"{p.name}. {p.description}" for p in products])
    points = [
        PointStruct(
            id=_point_id(p.id),
            vector=vectors[i].tolist(),
            payload={"id": p.id, "name": p.name, "description": p.description},
        )
        for i, p in enumerate(products)
    ]
    _client.upsert(collection_name=_collection(tenant), points=points)


def search(tenant: str, query: str, k: int):
    name = _collection(tenant)
    if not _client.collection_exists(name):
        return []
    qv = encode([query])[0].tolist()
    res = _client.query_points(collection_name=name, query=qv, limit=k, with_payload=True)
    return [point.payload["id"] for point in res.points]
