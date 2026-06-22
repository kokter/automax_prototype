"""Гибрид: объединение лексического и семантического ранжирований через RRF."""
from app import config


def rrf(lexical_ids: list[str], semantic_ids: list[str], k: int):
    """Reciprocal Rank Fusion. Возвращает [(id, score), ...] по убыванию."""
    scores: dict[str, float] = {}
    for rank, pid in enumerate(lexical_ids):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (config.RRF_K + rank)
    for rank, pid in enumerate(semantic_ids):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (config.RRF_K + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:k]
