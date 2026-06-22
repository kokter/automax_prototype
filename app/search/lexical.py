"""Лексический слой: морфология (леммы) + опечатки (fuzzy) + BM25.

Хранится в памяти процесса, отдельный индекс на каждого арендатора. В продакшене
этот слой заменяется на OpenSearch/Manticore; для MVP достаточно in-process BM25.
"""
import re
from functools import lru_cache
from threading import Lock

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


class LexicalStore:
    """Реестр лексических индексов по арендаторам."""
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


lexical_store = LexicalStore()
