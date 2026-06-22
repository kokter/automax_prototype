"""Загрузка демо-каталога в запущенный сервис (POST /index)."""
import json
import sys
import httpx

API = "http://localhost:8000"
API_KEY = "demo-key-automax"

catalog = json.load(open("sample_catalog.json", encoding="utf-8"))
try:
    r = httpx.post(f"{API}/index", json=catalog, headers={"X-API-Key": API_KEY}, timeout=600)
    r.raise_for_status()
    print("Проиндексировано:", r.json())
except Exception as e:
    print("Ошибка: убедитесь, что сервис запущен (uvicorn) и Qdrant поднят. Детали:", e)
    sys.exit(1)
