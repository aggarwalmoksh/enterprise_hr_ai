import json
import os
from datetime import datetime
from pathlib import Path

from app.utils.config import DATA_DIR
from app.utils.logger import get_logger

logger = get_logger("prediction_logger")

PREDICTIONS_DIR = DATA_DIR / "predictions"


def _ensure_dir():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _get_filename() -> str:
    return datetime.now().strftime("%Y-%m-%d") + ".jsonl"


def log_prediction(
    employee_id: int,
    model_version: str,
    probability: float,
    risk: str,
    department: str = None,
    extra: dict = None,
):
    _ensure_dir()

    record = {
        "timestamp": datetime.now().isoformat(),
        "employee_id": employee_id,
        "model_version": model_version,
        "probability": probability,
        "risk": risk,
    }
    if department:
        record["department"] = department
    if extra:
        record.update(extra)

    filepath = PREDICTIONS_DIR / _get_filename()
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Prediction logged | EmployeeID=%d, Risk=%s, Prob=%.4f, File=%s",
        employee_id, risk, probability, filepath.name,
    )


def log_batch_prediction(
    model_version: str,
    predictions: list[dict],
):
    _ensure_dir()

    filepath = PREDICTIONS_DIR / _get_filename()
    with open(filepath, "a") as f:
        for pred in predictions:
            emp_id = pred.get("employee_id") or pred.get("EmployeeID")
            record = {
                "timestamp": datetime.now().isoformat(),
                "employee_id": emp_id,
                "model_version": model_version,
                "probability": pred.get("attrition_probability", pred.get("probability")),
                "risk": pred["risk"],
            }
            f.write(json.dumps(record) + "\n")

    logger.info(
        "Batch prediction logged | Count=%d, File=%s",
        len(predictions), filepath.name,
    )


def load_predictions(date: str = None) -> list[dict]:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    filepath = PREDICTIONS_DIR / f"{date}.jsonl"
    if not filepath.exists():
        return []

    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_prediction_stats(date: str = None) -> dict:
    records = load_predictions(date)
    if not records:
        return {"count": 0, "high": 0, "medium": 0, "low": 0}

    risks = [r["risk"] for r in records]
    return {
        "count": len(records),
        "high": risks.count("HIGH"),
        "medium": risks.count("MEDIUM"),
        "low": risks.count("LOW"),
        "avg_probability": round(sum(r["probability"] for r in records) / len(records), 4),
    }
