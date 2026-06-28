import re
from functools import lru_cache
from threading import Lock

from app import config


# ---------------------------------------------------------------------------
# Запасной backend: BM25 в памяти (pymorphy3 + rank-bm25 + rapidfuzz)
# ---------------------------------------------------------------------------
import pymorphy3
from rank_bm25 import BM25Okapi
from rapidfuzz import process, fuzz

_morph = pymorphy3.MorphAnalyzer()
_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
_STOP = {"на", "в", "во", "по", "из", "для", "и", "с", "к", "у", "о", "об",
         "за", "под", "над", "при", "же", "бы", "а", "но", "или", "до"}


@lru_cache(maxsize=50000)
def _lemma(tok: str) -> str:
    return _morph.parse(tok)[0].normal_form


def _tokens(text: str):
    return [t.lower() for t in _WORD.findall(text) if t.lower() not in _STOP]


def _lemmas(text: str):
    return [_lemma(t) for t in _tokens(text)]


class _TenantIndex:
    def __init__(self):
        self.ids: list[str] = []
        self.doc_lemmas: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.vocab: list[str] = []

    def build(self, products):
        self.ids = [p.id for p in products]
        self.doc_lemmas = [_lemmas(f"{p.name}. {p.description}") for p in products]
        self.bm25 = BM25Okapi(self.doc_lemmas) if self.doc_lemmas else None
        self.vocab = sorted({l for dl in self.doc_lemmas for l in dl})

    def _expand(self, query: str):
        out = []
        for tok in _tokens(query):
            lm = _lemma(tok)
            if lm not in self.vocab and self.vocab:
                m = process.extractOne(lm, self.vocab, scorer=fuzz.ratio)
                if m and m[1] >= 80:
                    out.append(m[0])
                    continue
            out.append(lm)
        return out

    def search(self, query: str, k: int):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._expand(query))
        order = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in order if scores[i] > 0][:k]


class InMemoryBackend:
    """Запасной лексический индекс в оперативной памяти."""
    name = "in-memory"

    def __init__(self):
        self._by_tenant: dict[str, _TenantIndex] = {}
        self._lock = Lock()

    def build(self, tenant: str, products):
        idx = _TenantIndex()
        idx.build(products)
        with self._lock:
            self._by_tenant[tenant] = idx

    def search(self, tenant: str, query: str, k: int):
        idx = self._by_tenant.get(tenant)
        return idx.search(query, k) if idx else []


_RU_MAPPING = {
    "settings": {
        "analysis": {
            "filter": {
                "ru_stop": {"type": "stop", "stopwords": "_russian_"},
                "ru_stemmer": {"type": "stemmer", "language": "russian"},
            },
            "analyzer": {
                "ru": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "ru_stop", "ru_stemmer"],
                }
            },
        }
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "ru"},
            "description": {"type": "text", "analyzer": "ru"},
        }
    },
}


class OpenSearchBackend:
    """Лексический поиск на OpenSearch: индекс на арендатора, BM25, морфология, опечатки."""
    name = "opensearch"

    def __init__(self, url: str):
        from opensearchpy import OpenSearch  # ленивый импорт
        self._OpenSearch = OpenSearch
        self.client = OpenSearch(
            hosts=[url],
            http_compress=True,
            use_ssl=url.startswith("https"),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )

    def _index(self, tenant: str) -> str:
        # Имя индекса OpenSearch — в нижнем регистре, как и идентификатор арендатора.
        return f"{config.OPENSEARCH_INDEX_PREFIX}{tenant}".lower()

    def build(self, tenant: str, products):
        from opensearchpy.helpers import bulk
        index = self._index(tenant)
        # Переиндексация: пересоздаём индекс арендатора с нуля.
        if self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)
        self.client.indices.create(index=index, body=_RU_MAPPING)
        actions = [
            {
                "_index": index,
                "_id": p.id,
                "_source": {"id": p.id, "name": p.name, "description": p.description},
            }
            for p in products
        ]
        if actions:
            bulk(self.client, actions, refresh=True)

    def search(self, tenant: str, query: str, k: int):
        index = self._index(tenant)
        if not self.client.indices.exists(index=index):
            return []
        body = {
            "size": k,
            "_source": False,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["name^2", "description"],
                    "fuzziness": "AUTO",   # устойчивость к опечаткам
                    "operator": "or",
                }
            },
        }
        res = self.client.search(index=index, body=body)
        return [hit["_id"] for hit in res["hits"]["hits"]]


# ---------------------------------------------------------------------------
# Выбор backend по конфигурации (с безопасным фолбэком)
# ---------------------------------------------------------------------------
def _make_backend():
    url = config.OPENSEARCH_URL
    if url and url != ":memory:":
        try:
            backend = OpenSearchBackend(url)
            if backend.client.ping():
                return backend
            raise RuntimeError("ping() вернул False")
        except Exception as e:  # noqa: BLE001
            print(f"[lexical] OpenSearch недоступен по адресу {url} ({e}); "
                  f"использую запасной BM25-индекс в памяти.")
    return InMemoryBackend()


lexical_store = _make_backend()


def backend_name() -> str:
    """Активный лексический backend ('opensearch' или 'in-memory') — для /health."""
    return getattr(lexical_store, "name", "in-memory")
