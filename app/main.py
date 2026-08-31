from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.ml.model_loader import load_pipeline, load_metadata, get_model_version
from app.utils.logger import get_logger

logger = get_logger("app.main")

app = FastAPI(
    title="Enterprise HR AI",
    description="Attrition prediction, skill gap analysis, and upskilling recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    logger.info("Starting Enterprise HR AI service...")
    load_pipeline()
    load_metadata()
    logger.info("Service ready | Model=v%s", get_model_version())


@app.get("/")
def root():
    return {
        "service": "Enterprise HR AI",
        "version": "1.0.0",
        "model_version": get_model_version(),
        "endpoints": {
            "predict": "POST /predict/attrition",
            "summary": "GET /dashboard/summary",
            "by_department": "GET /dashboard/attrition-by-department",
            "skill_gaps": "GET /dashboard/skill-gaps",
            "recommendations": "GET /dashboard/recommendations",
            "employee": "GET /employees/{employee_id}",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": get_model_version()}
