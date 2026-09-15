import time

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from src.api.metrics import (
    RECOMMENDATION_LATENCY,
    RECOMMENDATION_REQUESTS,
    RECOMMENDATION_RESULTS,
    metrics_response,
)
from src.pipeline.recommend import RecommendationPipeline


app = FastAPI(
    title="Personalization & Ranking Intelligence Engine",
    version="1.0.0",
)

pipeline = RecommendationPipeline()


class RecommendationRequest(BaseModel):
    visitorid: int
    k: int = Field(default=10, ge=1, le=100)
    query: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "personalization-ranking-engine",
    }


@app.get("/metrics")
def metrics():
    body, content_type = metrics_response()
    return Response(
        content=body,
        media_type=content_type,
    )


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    endpoint = "/recommend"
    start_time = time.perf_counter()

    try:
        recommendations = pipeline.recommend(
            visitorid=request.visitorid,
            k=request.k,
            query=request.query,
        )

        RECOMMENDATION_REQUESTS.labels(
            endpoint=endpoint,
            status="200",
        ).inc()

        RECOMMENDATION_RESULTS.observe(len(recommendations))

        return {
            "visitorid": request.visitorid,
            "recommendations": recommendations,
        }

    except ValueError as exc:
        RECOMMENDATION_REQUESTS.labels(
            endpoint=endpoint,
            status="404",
        ).inc()

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception:
        RECOMMENDATION_REQUESTS.labels(
            endpoint=endpoint,
            status="500",
        ).inc()

        raise

    finally:
        RECOMMENDATION_LATENCY.labels(
            endpoint=endpoint,
        ).observe(time.perf_counter() - start_time)