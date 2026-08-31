import pandas as pd
import numpy as np

from app.ml.model_loader import load_pipeline, get_feature_columns, get_model_version
from app.utils.logger import get_logger

logger = get_logger("predictor")


def assign_risk(prob: float) -> str:
    if prob >= 0.6:
        return "HIGH"
    elif prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "Income_per_year" not in result.columns and "MonthlySalary" in result.columns:
        result["Income_per_year"] = result["MonthlySalary"] * 12

    if "Gap_since_promotion" not in result.columns and "LastPromotionYear" in result.columns:
        result["Gap_since_promotion"] = 2024 - result["LastPromotionYear"]

    if "Satisfaction_score" not in result.columns:
        if "CustomerSatisfaction" in result.columns and "WorkLifeBalanceScore" in result.columns:
            result["Satisfaction_score"] = (
                result["CustomerSatisfaction"] * 0.5
                + result["WorkLifeBalanceScore"] * 0.5
            )

    if "Experience_ratio" not in result.columns:
        if "YearsAtCompany" in result.columns and "Age" in result.columns:
            result["Experience_ratio"] = result["YearsAtCompany"] / result["Age"].clip(lower=1)

    return result


def predict_attrition(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = load_pipeline()
    feature_cols = get_feature_columns()

    df_engineered = _engineer_features(df)

    missing = [c for c in feature_cols if c not in df_engineered.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df_engineered[feature_cols].copy()
    probs = pipeline.predict_proba(X)[:, 1]

    result = df[["EmployeeID"]].copy() if "EmployeeID" in df.columns else pd.DataFrame()
    result["attrition_probability"] = np.round(probs, 4)
    result["risk"] = result["attrition_probability"].apply(assign_risk)

    high = int((result["risk"] == "HIGH").sum())
    medium = int((result["risk"] == "MEDIUM").sum())
    low = int((result["risk"] == "LOW").sum())

    logger.info(
        "Batch prediction completed: %d employees | HIGH=%d, MEDIUM=%d, LOW=%d",
        len(result), high, medium, low,
    )
    return result


def predict_single(employee_data: dict) -> dict:
    pipeline = load_pipeline()
    feature_cols = get_feature_columns()

    df = pd.DataFrame([employee_data])
    df_engineered = _engineer_features(df)

    missing = [c for c in feature_cols if c not in df_engineered.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df_engineered[feature_cols]
    prob = pipeline.predict_proba(X)[0, 1]
    risk = assign_risk(prob)

    return {
        "attrition_probability": round(float(prob), 4),
        "risk": risk,
    }
