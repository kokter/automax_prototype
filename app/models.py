"""Схемы запросов и ответов API."""
from typing import List
from pydantic import BaseModel, Field


class Product(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор товара")
    name: str = Field(..., description="Название товара")
    description: str = Field("", description="Описание/назначение товара")


class IndexRequest(BaseModel):
    products: List[Product]


class IndexResponse(BaseModel):
    tenant: str
    indexed: int


class SearchHit(BaseModel):
    id: str
    name: str
    description: str
    score: float


class SearchResponse(BaseModel):
    query: str
    tenant: str
    hits: List[SearchHit]
