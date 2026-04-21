"""Root entry point for the Wave interactive web app.

The FastAPI application lives at `app/backend/main.py`; this file is a
launcher so the repository layout matches the course-required structure
(a top-level `app.py`).

Usage
-----
    python app.py                            # launch on http://localhost:8000
    uvicorn app.backend.main:app --reload    # equivalent, with hot-reload

Production (Railway) uses `uvicorn app.backend.main:app` directly via the
Dockerfile CMD, so this launcher only affects local development ergonomics.
"""

from __future__ import annotations


def main() -> None:
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
