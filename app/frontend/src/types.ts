export type Modality = "book" | "film" | "music" | "writing";
export type ModelName = "two_tower" | "knn" | "popularity";

export interface ProductCard {
  id: string;
  modality: string;
  title: string;
  creator: string;
  year: number;
  cover_url: string;
  external_url: string;
  similarity: number;
  why_this: string;
  // Sub-type within modality: "poem" | "essay" | "article" | ""
  subtype?: string;
  // First ~400 chars of raw description (poem body / essay summary / article lede).
  excerpt?: string;
}

export interface QueryProfile {
  vibe_summary: string;
  mood_vector: number[];
  intent_vector: number[];
  aesthetic_tags: string[];
}

export interface RecommendResponse {
  query_profile: QueryProfile;
  results: Record<string, ProductCard[]>;
}

export interface RecommendRequest {
  query: string;
  modalities?: Modality[];
  model?: ModelName;
  image_base64?: string; // full data URL: "data:image/jpeg;base64,..."
  bypass_cache?: boolean;
}

export interface RecommendAllRequest {
  query: string;
  modalities?: Modality[];
}

export interface RecommendAllResponse {
  query_profile: QueryProfile;
  // Outer key: model name; inner key: modality; value: list of cards.
  results_by_model: Record<ModelName, Record<string, ProductCard[]>>;
}
