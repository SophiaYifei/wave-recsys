"""Pydantic request/response schemas for the Wave API (spec §5.7)."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryProfile(BaseModel):
    vibe_summary: str
    mood_vector: List[float]
    intent_vector: List[float]
    aesthetic_tags: List[str]


class ProductCard(BaseModel):
    id: str
    modality: str
    title: str
    creator: str
    year: int
    cover_url: str
    external_url: str
    similarity: float
    why_this: str


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    modalities: Optional[List[str]] = None  # None -> all four
    model: str = Field(default="two_tower")  # "popularity" | "knn" | "two_tower"


class RecommendResponse(BaseModel):
    query_profile: QueryProfile
    results: Dict[str, List[ProductCard]]
