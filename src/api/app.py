from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.recommend import RecommendationPipeline


app = FastAPI(
    title="Personalization & Ranking Intelligence Engine",
    version="1.0.0",
)


pipeline = RecommendationPipeline()


class RecommendationRequest(BaseModel):
    visitorid: int
    k: int = Field(default=10, ge=1, le=100)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "personalization-ranking-engine",
    }


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        recommendations = pipeline.recommend(
            visitorid=request.visitorid,
            k=request.k,
        )

        return {
            "visitorid": request.visitorid,
            "recommendations": recommendations,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
