import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestRootEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "Enterprise HR AI"
        assert "endpoints" in data

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


class TestPredictAttrition:
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

    def test_valid_prediction(self):
        r = client.post("/predict/attrition", json=self._valid_input())
        assert r.status_code == 200
        data = r.json()
        assert "attrition_probability" in data
        assert "risk" in data
        assert data["risk"] in ["HIGH", "MEDIUM", "LOW"]
        assert 0 <= data["attrition_probability"] <= 1

    def test_invalid_department(self):
        data = self._valid_input()
        data["Department"] = "InvalidDept"
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422

    def test_missing_field(self):
        data = self._valid_input()
        del data["Department"]
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422

    def test_invalid_age(self):
        data = self._valid_input()
        data["Age"] = 10
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422

    def test_invalid_salary(self):
        data = self._valid_input()
        data["MonthlySalary"] = -100
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422

    def test_invalid_performance_rating(self):
        data = self._valid_input()
        data["PerformanceRating"] = 6
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422

    def test_empty_body(self):
        r = client.post("/predict/attrition", json={})
        assert r.status_code == 422

    def test_missing_education_level(self):
        data = self._valid_input()
        del data["EducationLevel"]
        r = client.post("/predict/attrition", json=data)
        assert r.status_code == 422


class TestDashboardEndpoints:
    def test_summary(self):
        r = client.get("/dashboard/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_employees" in data
        assert "high_risk_employees" in data
        assert "average_engagement" in data
        assert data["total_employees"] > 0

    def test_attrition_by_department(self):
        r = client.get("/dashboard/attrition-by-department")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "Department" in data[0]
        assert "employee_count" in data[0]

    def test_skill_gaps(self):
        r = client.get("/dashboard/skill-gaps")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "skill" in data[0]
        assert "severity" in data[0]

    def test_recommendations(self):
        r = client.get("/dashboard/recommendations")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)


class TestEmployeeEndpoint:
    def test_valid_employee(self):
        r = client.get("/employees/1")
        assert r.status_code == 200
        data = r.json()
        assert data["employee_id"] == 1
        assert "department" in data
        assert "attrition" in data
        assert "skill_gaps" in data
        assert "recommendations" in data

    def test_nonexistent_employee(self):
        r = client.get("/employees/99999")
        assert r.status_code == 404

    def test_invalid_employee_id(self):
        r = client.get("/employees/0")
        assert r.status_code == 400

    def test_negative_employee_id(self):
        r = client.get("/employees/-1")
        assert r.status_code == 400
