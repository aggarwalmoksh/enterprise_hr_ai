import pandas as pd
from fastapi import APIRouter, HTTPException

from app.ml.predictor import predict_single, predict_attrition
from app.ml.model_loader import get_model_version
from app.ml.prediction_logger import log_prediction, log_batch_prediction
from app.services import skill_gap_service, recommendation_service
from app.api.schemas import EmployeeInput, PredictionResponse
from app.utils.config import EMPLOYEE_FILE, ENGAGEMENT_FILE
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger("api.endpoints")


@router.post(
    "/predict/attrition",
    response_model=PredictionResponse,
    responses={
        400: {"description": "Validation error"},
        422: {"description": "Unprocessable entity"},
    },
)
def predict_attrition_single(emp: EmployeeInput):
    logger.info(
        "Prediction request received | EmployeeID=%d, Department=%s",
        emp.employee_id, emp.department,
    )

    try:
        pred = predict_single(emp.model_dump(by_alias=True))
    except ValueError as e:
        logger.warning(
            "Prediction failed | EmployeeID=%d, Error=%s",
            emp.employee_id, str(e),
        )
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        logger.error(
            "Unexpected error | EmployeeID=%d, Error=%s",
            emp.employee_id, str(e),
        )
        raise HTTPException(status_code=500, detail="Internal prediction error")

    log_prediction(
        employee_id=emp.employee_id,
        model_version=get_model_version(),
        probability=pred["attrition_probability"],
        risk=pred["risk"],
        department=emp.department,
    )

    logger.info(
        "Prediction completed | EmployeeID=%d, Risk=%s, Probability=%.4f, Model=v%s",
        emp.employee_id, pred["risk"], pred["attrition_probability"], get_model_version(),
    )

    return PredictionResponse(
        employee_id=emp.employee_id,
        attrition_probability=pred["attrition_probability"],
        risk=pred["risk"],
        message=f"Prediction complete for employee {emp.employee_id}",
    )


@router.get("/dashboard/summary")
def dashboard_summary():
    logger.info("Dashboard summary requested")
    df = pd.read_csv(EMPLOYEE_FILE)
    preds = predict_attrition(df)

    log_batch_prediction(
        model_version=get_model_version(),
        predictions=preds.to_dict(orient="records"),
    )

    eng = pd.read_csv(ENGAGEMENT_FILE)
    avg_engagement = round(float(eng["Engagement Score"].mean()), 2)

    result = {
        "total_employees": len(df),
        "high_risk_employees": int((preds["risk"] == "HIGH").sum()),
        "average_engagement": avg_engagement,
    }
    logger.info("Dashboard summary returned: %d total, %d high-risk", result["total_employees"], result["high_risk_employees"])
    return result


@router.get("/dashboard/attrition-by-department")
def attrition_by_department():
    logger.info("Attrition by department requested")
    df = pd.read_csv(EMPLOYEE_FILE)
    preds = predict_attrition(df)

    merged = df[["EmployeeID", "Department"]].merge(preds, on="EmployeeID")

    dept_stats = merged.groupby("Department").agg(
        employee_count=("EmployeeID", "count"),
        high_risk_count=("risk", lambda x: (x == "HIGH").sum()),
        avg_probability=("attrition_probability", "mean"),
    ).round(4).reset_index()

    return dept_stats.to_dict(orient="records")


@router.get("/dashboard/skill-gaps")
def skill_gaps():
    logger.info("Organization skill gaps requested")
    gaps = skill_gap_service.get_org_wide_gaps()
    logger.info("Skill gaps returned: %d unique skills", len(gaps))
    return gaps.to_dict(orient="records")


@router.get("/dashboard/recommendations")
def recommendations():
    logger.info("Training recommendations requested")
    recs = recommendation_service.get_employee_recommendations()
    logger.info("Recommendations returned: %d employees", len(recs))
    return recs.to_dict(orient="records")


@router.get("/employees/{employee_id}")
def employee_intelligence(employee_id: int):
    logger.info("Employee intelligence requested | EmployeeID=%d", employee_id)

    if employee_id < 1:
        raise HTTPException(status_code=400, detail="Employee ID must be positive")

    df = pd.read_csv(EMPLOYEE_FILE)
    emp = df[df["EmployeeID"] == employee_id]
    if emp.empty:
        logger.warning("Employee not found | EmployeeID=%d", employee_id)
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    pred = predict_single(emp.iloc[0].to_dict())

    emp_gaps = skill_gap_service.get_employee_gaps()
    emp_gap = emp_gaps[emp_gaps["employee_id"] == employee_id]
    skill_gaps = emp_gap.iloc[0]["skill_gaps"] if not emp_gap.empty else []

    recs = recommendation_service.get_recommendations(skill_gaps)

    emp_data = emp.iloc[0]
    logger.info(
        "Employee intelligence returned | EmployeeID=%d, Risk=%s, Gaps=%d, Recs=%d",
        employee_id, pred["risk"], len(skill_gaps), len(recs),
    )
    return {
        "employee_id": employee_id,
        "department": emp_data["Department"],
        "job_role": emp_data["JobRole"],
        "age": int(emp_data["Age"]),
        "gender": emp_data["Gender"],
        "years_at_company": int(emp_data["YearsAtCompany"]),
        "monthly_salary": float(emp_data["MonthlySalary"]),
        "attrition": pred,
        "skill_gaps": skill_gaps,
        "recommendations": recs,
    }
