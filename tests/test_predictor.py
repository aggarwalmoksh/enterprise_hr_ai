import pytest
import pandas as pd
import numpy as np

from app.ml.predictor import assign_risk, _engineer_features, predict_single, predict_attrition
from app.ml.model_loader import load_pipeline, load_metadata, get_feature_columns


class TestAssignRisk:
    def test_high_risk(self):
        assert assign_risk(0.8) == "HIGH"
        assert assign_risk(1.0) == "HIGH"

    def test_medium_risk(self):
        assert assign_risk(0.3) == "MEDIUM"
        assert assign_risk(0.5) == "MEDIUM"
        assert assign_risk(0.59) == "MEDIUM"

    def test_low_risk(self):
        assert assign_risk(0.0) == "LOW"
        assert assign_risk(0.29) == "LOW"

    def test_boundary_high_medium(self):
        assert assign_risk(0.6) == "HIGH"
        assert assign_risk(0.5999) == "MEDIUM"

    def test_boundary_medium_low(self):
        assert assign_risk(0.3) == "MEDIUM"
        assert assign_risk(0.2999) == "LOW"


class TestEngineerFeatures:
    def test_income_per_year(self):
        df = pd.DataFrame({"MonthlySalary": [5000, 10000]})
        result = _engineer_features(df)
        assert "Income_per_year" in result.columns
        assert result["Income_per_year"].tolist() == [60000, 120000]

    def test_gap_since_promotion(self):
        df = pd.DataFrame({"LastPromotionYear": [2020, 2023]})
        result = _engineer_features(df)
        assert "Gap_since_promotion" in result.columns
        assert result["Gap_since_promotion"].tolist() == [4, 1]

    def test_satisfaction_score(self):
        df = pd.DataFrame({
            "CustomerSatisfaction": [8.0, 6.0],
            "WorkLifeBalanceScore": [7.0, 5.0],
        })
        result = _engineer_features(df)
        assert "Satisfaction_score" in result.columns
        assert result["Satisfaction_score"].iloc[0] == pytest.approx(7.5)
        assert result["Satisfaction_score"].iloc[1] == pytest.approx(5.5)

    def test_experience_ratio(self):
        df = pd.DataFrame({"YearsAtCompany": [5, 10], "Age": [30, 40]})
        result = _engineer_features(df)
        assert "Experience_ratio" in result.columns
        assert result["Experience_ratio"].iloc[0] == pytest.approx(5 / 30)
        assert result["Experience_ratio"].iloc[1] == pytest.approx(10 / 40)

    def test_missing_columns_handled(self):
        df = pd.DataFrame({"SomeOtherCol": [1, 2]})
        result = _engineer_features(df)
        assert "Income_per_year" not in result.columns
        assert "Gap_since_promotion" not in result.columns


class TestPredictSingle:
    def _valid_input(self):
        return {
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

    def test_returns_probability(self):
        result = predict_single(self._valid_input())
        assert "attrition_probability" in result
        assert isinstance(result["attrition_probability"], float)
        assert 0 <= result["attrition_probability"] <= 1

    def test_returns_risk(self):
        result = predict_single(self._valid_input())
        assert "risk" in result
        assert result["risk"] in ["HIGH", "MEDIUM", "LOW"]

    def test_missing_column_raises(self):
        data = self._valid_input()
        del data["Department"]
        with pytest.raises(ValueError, match="Missing feature columns"):
            predict_single(data)

    def test_probability_is_between_0_and_1(self):
        result = predict_single(self._valid_input())
        assert 0.0 <= result["attrition_probability"] <= 1.0


class TestPredictAttrition:
    def test_batch_prediction(self):
        load_pipeline()
        load_metadata()

        df = pd.DataFrame({
            "EmployeeID": [1, 2, 3],
            "Department": ["IT", "Sales", "Finance"],
            "JobRole": ["Developer", "Executive", "Analyst"],
            "Age": [30, 45, 35],
            "Gender": ["Male", "Female", "Male"],
            "YearsAtCompany": [5, 10, 3],
            "MonthlySalary": [8000, 12000, 7000],
            "PerformanceRating": [4, 5, 3],
            "OvertimeHoursPerMonth": [10, 5, 15],
            "LeavesTaken": [5, 3, 8],
            "ProjectsHandled": [8, 12, 6],
            "TrainingHours": [40, 60, 30],
            "CustomerSatisfaction": [7.5, 8.0, 6.5],
            "LastPromotionYear": [2022, 2021, 2023],
            "WorkLifeBalanceScore": [6.0, 7.0, 5.5],
            "Country": ["USA", "UK", "India"],
            "LeaveDayName": ["Monday", "Tuesday", "Wednesday"],
            "EducationLevel": ["Bachelor", "Master", "Bachelor"],
        })

        result = predict_attrition(df)
        assert len(result) == 3
        assert "attrition_probability" in result.columns
        assert "risk" in result.columns
        assert all(0 <= p <= 1 for p in result["attrition_probability"])
        assert all(r in ["HIGH", "MEDIUM", "LOW"] for r in result["risk"])
