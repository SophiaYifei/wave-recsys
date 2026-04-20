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
    # Sub-type within modality (writing -> "poem" | "essay" | "article"; others -> "")
    subtype: str = ""
    # First ~400 chars of the raw description, useful as a textual poster when
    # cover_url is missing (PoetryDB poems) or for a quick preview in tooltips.
    excerpt: str = ""


class RecommendRequest(BaseModel):
    # Empty string allowed when an image alone is the input; the endpoint
    # still validates that at least one of (query, image_base64) is non-empty.
    query: str = Field(default="", max_length=1000)
    modalities: Optional[List[str]] = None  # None -> all four
    model: str = Field(default="two_tower")  # "popularity" | "knn" | "two_tower"
    bypass_cache: bool = False  # set true to force a fresh LLM run
    # Full data URL, e.g. "data:image/jpeg;base64,/9j/...". Optional.
    # Limit the encoded payload size to avoid oversized request bodies.
    image_base64: Optional[str] = Field(default=None, max_length=7_000_000)


class RecommendResponse(BaseModel):
    query_profile: QueryProfile
    results: Dict[str, List[ProductCard]]


class RecommendAllRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    modalities: Optional[List[str]] = None  # None -> all four


class RecommendAllResponse(BaseModel):
    query_profile: QueryProfile
    # Outer key is model name (two_tower / knn / popularity).
    # Inner key is modality (book / film / music / writing).
    results_by_model: Dict[str, Dict[str, List[ProductCard]]]
