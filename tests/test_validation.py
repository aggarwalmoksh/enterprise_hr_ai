import pytest
from pydantic import ValidationError

from app.api.schemas import EmployeeInput


class TestEmployeeInputValidation:
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

    def test_valid_input(self):
        emp = EmployeeInput(**self._valid_input())
        assert emp.employee_id == 101
        assert emp.department == "IT"

    def test_missing_required_field(self):
        data = self._valid_input()
        del data["Department"]
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "Department" in str(exc_info.value)

    def test_invalid_department(self):
        data = self._valid_input()
        data["Department"] = "InvalidDept"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "Department" in str(exc_info.value)

    def test_age_too_low(self):
        data = self._valid_input()
        data["Age"] = 10
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "Age" in str(exc_info.value)

    def test_age_too_high(self):
        data = self._valid_input()
        data["Age"] = 105
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "Age" in str(exc_info.value)

    def test_invalid_gender(self):
        data = self._valid_input()
        data["Gender"] = "Other"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "Gender" in str(exc_info.value)

    def test_negative_salary(self):
        data = self._valid_input()
        data["MonthlySalary"] = -100
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "MonthlySalary" in str(exc_info.value)

    def test_salary_too_high(self):
        data = self._valid_input()
        data["MonthlySalary"] = 1000000
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "MonthlySalary" in str(exc_info.value)

    def test_performance_rating_too_high(self):
        data = self._valid_input()
        data["PerformanceRating"] = 6
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "PerformanceRating" in str(exc_info.value)

    def test_performance_rating_too_low(self):
        data = self._valid_input()
        data["PerformanceRating"] = 0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "PerformanceRating" in str(exc_info.value)

    def test_invalid_education_level(self):
        data = self._valid_input()
        data["EducationLevel"] = "Associate"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "EducationLevel" in str(exc_info.value)

    def test_negative_overtime(self):
        data = self._valid_input()
        data["OvertimeHoursPerMonth"] = -5
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "OvertimeHoursPerMonth" in str(exc_info.value)

    def test_work_life_balance_out_of_range(self):
        data = self._valid_input()
        data["WorkLifeBalanceScore"] = 15
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "WorkLifeBalanceScore" in str(exc_info.value)

    def test_customer_satisfaction_out_of_range(self):
        data = self._valid_input()
        data["CustomerSatisfaction"] = -1
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "CustomerSatisfaction" in str(exc_info.value)

    def test_empty_job_role(self):
        data = self._valid_input()
        data["JobRole"] = ""
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**data)
        assert "JobRole" in str(exc_info.value)
