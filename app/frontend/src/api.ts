import type {
  RecommendAllRequest,
  RecommendAllResponse,
  RecommendRequest,
  RecommendResponse,
  SwapRequest,
  SwapResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export async function getRecommendations(
  params: RecommendRequest,
): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let detail = `API ${res.status}`;
    try {
      const j = await res.json();
      detail += `: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch {
      detail += `: ${await res.text()}`;
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getAllRecommendations(
  params: RecommendAllRequest,
): Promise<RecommendAllResponse> {
  const res = await fetch(`${API_BASE}/api/recommend_all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let detail = `API ${res.status}`;
    try {
      const j = await res.json();
      detail += `: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch {
      detail += `: ${await res.text()}`;
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function swapCard(params: SwapRequest): Promise<SwapResponse> {
  const res = await fetch(`${API_BASE}/api/swap`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    let detail = `API ${res.status}`;
    try {
      const j = await res.json();
      detail += `: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`;
    } catch {
      detail += `: ${await res.text()}`;
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
