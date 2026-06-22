"""Прототип SaaS-сервиса гибридного поиска для интернет-магазина автозапчастей.

Эндпоинты:
  POST /index   — проиндексировать каталог арендатора (по API-ключу)
  GET  /search  — гибридный поиск (лексика + семантика, объединение RRF)
  GET  /        — простая веб-страница для теста поиска
  GET  /health  — проверка живости
"""
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from app import config
from app.models import IndexRequest, IndexResponse, SearchResponse, SearchHit
from app.search import lexical, semantic, hybrid

app = FastAPI(title="Automax Hybrid Search (MVP)", version="0.1.0")

_catalog: dict[str, dict[str, dict]] = {}


def resolve_tenant(api_key: str | None) -> str:
    tenant = config.API_KEYS.get(api_key or "")
    if not tenant:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий API-ключ (заголовок X-API-Key)")
    return tenant


@app.get("/health")
def health():
    return {"status": "ok", "model": config.EMBED_MODEL, "qdrant": config.QDRANT_URL}


@app.post("/index", response_model=IndexResponse)
def index_catalog(req: IndexRequest, x_api_key: str | None = Header(default=None)):
    tenant = resolve_tenant(x_api_key)
    products = req.products
    lexical.lexical_store.build(tenant, products)
    semantic.index(tenant, products)
    _catalog[tenant] = {p.id: {"name": p.name, "description": p.description} for p in products}
    return IndexResponse(tenant=tenant, indexed=len(products))


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), x_api_key: str | None = Header(default=None),
           k: int = Query(default=5, ge=1, le=50)):
    tenant = resolve_tenant(x_api_key)
    lex_ids = lexical.lexical_store.search(tenant, q, config.CANDIDATES)
    sem_ids = semantic.search(tenant, q, config.CANDIDATES)
    fused = hybrid.rrf(lex_ids, sem_ids, k)

    store = _catalog.get(tenant, {})
    hits = []
    for pid, score in fused:
        meta = store.get(pid, {"name": pid, "description": ""})
        hits.append(SearchHit(id=pid, name=meta["name"], description=meta["description"], score=round(score, 5)))
    return SearchResponse(query=q, tenant=tenant, hits=hits)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(WEB_PAGE)


WEB_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Автомакс · гибридный поиск (MVP)</title>
<style>
  body { font-family: system-ui, Arial, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; color: #1c2430; }
  h1 { font-size: 20px; color: #1F4E79; }
  .row { display: flex; gap: 8px; margin: 14px 0; }
  input { padding: 10px 12px; border: 1px solid #c8d2dd; border-radius: 8px; font-size: 15px; }
  #q { flex: 1; }
  #key { width: 200px; }
  button { padding: 10px 16px; border: 0; border-radius: 8px; background: #2E75B6; color: #fff; font-size: 15px; cursor: pointer; }
  .hint { color: #7a8794; font-size: 13px; }
  .hit { border: 1px solid #e6ebf1; border-radius: 10px; padding: 12px 14px; margin: 10px 0; }
  .hit .name { font-weight: 600; }
  .hit .desc { color: #5a6776; font-size: 14px; margin-top: 4px; }
  .hit .score { color: #98a3b0; font-size: 12px; margin-top: 6px; }
  .empty { color: #9aa5b1; margin-top: 16px; }
</style></head>
<body>
  <h1>Автомакс · гибридный поиск товаров (прототип)</h1>
  <p class="hint">Лексика (морфология + опечатки + BM25) и семантика (bge-m3) объединяются через RRF.
  Попробуйте сленг и опечатки: «граната», «незамерзайка», «помпа», «аммортизатор».</p>
  <div class="row">
    <input id="q" placeholder="Введите запрос…" autofocus>
    <button onclick="run()">Найти</button>
  </div>
  <div class="row">
    <input id="key" value="demo-key-automax" title="API-ключ арендатора">
    <span class="hint" style="align-self:center">← API-ключ (мультитенантность)</span>
  </div>
  <div id="results"></div>
<script>
async function run() {
  const q = document.getElementById('q').value.trim();
  const key = document.getElementById('key').value.trim();
  const box = document.getElementById('results');
  if (!q) { box.innerHTML = ''; return; }
  box.innerHTML = '<div class="empty">Поиск…</div>';
  try {
    const r = await fetch('/search?q=' + encodeURIComponent(q) + '&k=5', { headers: { 'X-API-Key': key } });
    if (!r.ok) { box.innerHTML = '<div class="empty">Ошибка: ' + r.status + '</div>'; return; }
    const data = await r.json();
    if (!data.hits.length) { box.innerHTML = '<div class="empty">Ничего не найдено.</div>'; return; }
    box.innerHTML = data.hits.map(h =>
      '<div class="hit"><div class="name">' + h.name + '</div>' +
      (h.description ? '<div class="desc">' + h.description + '</div>' : '') +
      '<div class="score">score: ' + h.score + '</div></div>').join('');
  } catch (e) { box.innerHTML = '<div class="empty">Сбой запроса.</div>'; }
}
document.getElementById('q').addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
</script>
</body></html>"""
