"""Конфигурация прототипа: модель, Qdrant, API-ключи арендаторов (мультитенантность)."""
import os

# Qdrant: по умолчанию — Docker. Для быстрых тестов можно ":memory:".
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Embedding-модель. Меняется через переменную окружения.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))  # bge-m3 -> 1024

# Тестовый режим без загрузки нейромодели (детерминированные псевдо-векторы).
# В реальном запуске НЕ включать — нужен настоящий bge-m3.
FAKE_EMBED = os.getenv("PROTOTYPE_FAKE_EMBED", "0") == "1"

# Мультитенантность: API-ключ -> идентификатор арендатора (магазина).
# Данные каждого арендатора изолированы (отдельная коллекция Qdrant и свой лексический индекс).
API_KEYS = {
    "demo-key-automax": "automax",
    "demo-key-second": "second_shop",
}

# Сколько кандидатов берём из каждого движка перед объединением.
CANDIDATES = int(os.getenv("CANDIDATES", "20"))
# Константа сглаживания RRF.
RRF_K = int(os.getenv("RRF_K", "60"))
