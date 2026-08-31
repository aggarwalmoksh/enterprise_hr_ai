from pydantic import BaseModel, Field
from typing import Literal


class EmployeeInput(BaseModel):
    employee_id: int = Field(
        ..., alias="EmployeeID", description="Unique employee identifier", ge=1
    )
    department: Literal["IT", "Sales", "Support", "Finance", "HR", "Marketing"] = Field(
        ..., alias="Department"
    )
    job_role: str = Field(..., alias="JobRole", min_length=1, max_length=100)
    age: int = Field(..., alias="Age", ge=18, le=100, description="Employee age")
    gender: Literal["Male", "Female", "Non-binary"] = Field(..., alias="Gender")
    years_at_company: int = Field(
        ..., alias="YearsAtCompany", ge=0, le=50, description="Tenure in years"
    )
    monthly_salary: float = Field(
        ..., alias="MonthlySalary", gt=0, le=500000, description="Monthly salary"
    )
    performance_rating: int = Field(
        ..., alias="PerformanceRating", ge=1, le=5, description="Rating 1-5"
    )
    overtime_hours_per_month: float = Field(
        ..., alias="OvertimeHoursPerMonth", ge=0, le=200
    )
    leaves_taken: int = Field(..., alias="LeavesTaken", ge=0, le=365)
    projects_handled: int = Field(..., alias="ProjectsHandled", ge=0, le=100)
    training_hours: float = Field(..., alias="TrainingHours", ge=0, le=1000)
    customer_satisfaction: float = Field(
        ..., alias="CustomerSatisfaction", ge=0, le=10
    )
    last_promotion_year: int = Field(
        ..., alias="LastPromotionYear", ge=2000, le=2025
    )
    work_life_balance_score: float = Field(
        ..., alias="WorkLifeBalanceScore", ge=0, le=10
    )
    country: str = Field(..., alias="Country", min_length=1, max_length=100)
    leave_day_name: str = Field(..., alias="LeaveDayName", min_length=1, max_length=20)
    education_level: Literal["High School", "Bachelor", "Master", "PhD"] = Field(
        ..., alias="EducationLevel"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "EmployeeID": 101,
                "Department": "IT",
                "JobRole": "Data Analyst",
                "Age": 30,
                "Gender": "Male",
                "YearsAtCompany": 5,
                "MonthlySalary": 8000,
                "PerformanceRating": 4,
                "OvertimeHoursPerMonth": 10,
                "LeavesTaken": 5,
                "ProjectsHandled": 8,
                "TrainingHours": 40,
                "CustomerSatisfaction": 7.5,
                "LastPromotionYear": 2022,
                "WorkLifeBalanceScore": 6.0,
                "Country": "USA",
                "LeaveDayName": "Monday",
                "EducationLevel": "Bachelor",
            }
        }


class AttritionPrediction(BaseModel):
    employee_id: int
    attrition_probability: float = Field(..., ge=0, le=1)
    risk: Literal["HIGH", "MEDIUM", "LOW"]


class PredictionResponse(BaseModel):
    employee_id: int
    attrition_probability: float
    risk: str
    message: str
