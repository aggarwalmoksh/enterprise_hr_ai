import pytest
import pandas as pd

from app.services.skill_gap_service import (
    _safe_parse_list,
    _map_to_department,
    get_employee_gaps,
    get_org_wide_gaps,
    get_department_gaps,
)


class TestSafeParseList:
    def test_valid_list_string(self):
        assert _safe_parse_list("['a', 'b', 'c']") == ["a", "b", "c"]

    def test_empty_list_string(self):
        assert _safe_parse_list("[]") == []

    def test_actual_list(self):
        assert _safe_parse_list(["a", "b"]) == ["a", "b"]

    def test_invalid_string(self):
        assert _safe_parse_list("not a list") == []

    def test_non_string_non_list(self):
        assert _safe_parse_list(123) == []

    def test_none(self):
        assert _safe_parse_list(None) == []


class TestMapToDepartment:
    def test_it_department(self):
        assert _map_to_department("Software Developers") == "IT"
        assert _map_to_department("Computer Systems Analysts") == "IT"
        assert _map_to_department("Database Administrators") == "IT"

    def test_sales_department(self):
        assert _map_to_department("Sales Managers") == "Sales"
        assert _map_to_department("Advertising Managers") == "Sales"

    def test_support_department(self):
        assert _map_to_department("Help Desk Technicians") == "Support"
        assert _map_to_department("Customer Support Specialists") == "Support"

    def test_finance_department(self):
        assert _map_to_department("Financial Managers") == "Finance"
        assert _map_to_department("Financial Analysts") == "Finance"

    def test_hr_department(self):
        assert _map_to_department("Training and Development Specialists") == "HR"

    def test_nan_returns_other(self):
        assert _map_to_department(float("nan")) == "Other"


class TestEmployeeGaps:
    def test_returns_dataframe(self):
        gaps = get_employee_gaps()
        assert isinstance(gaps, pd.DataFrame)
        assert len(gaps) == 500

    def test_has_required_columns(self):
        gaps = get_employee_gaps()
        assert "employee_id" in gaps.columns
        assert "department" in gaps.columns
        assert "gap_count" in gaps.columns
        assert "skill_gaps" in gaps.columns

    def test_gap_count_is_non_negative(self):
        gaps = get_employee_gaps()
        assert all(g >= 0 for g in gaps["gap_count"])

    def test_skill_gaps_are_lists(self):
        gaps = get_employee_gaps()
        for _, row in gaps.head(10).iterrows():
            assert isinstance(row["skill_gaps"], list)


class TestOrgWideGaps:
    def test_returns_dataframe(self):
        gaps = get_org_wide_gaps()
        assert isinstance(gaps, pd.DataFrame)
        assert len(gaps) > 0

    def test_has_severity_column(self):
        gaps = get_org_wide_gaps()
        assert "severity" in gaps.columns

    def test_severity_values(self):
        gaps = get_org_wide_gaps()
        valid_severities = {"HIGH", "MEDIUM", "LOW"}
        assert all(s in valid_severities for s in gaps["severity"])

    def test_sorted_by_count(self):
        gaps = get_org_wide_gaps()
        counts = gaps["employees_missing"].tolist()
        assert counts == sorted(counts, reverse=True)


class TestDepartmentGaps:
    def test_returns_dataframe(self):
        gaps = get_department_gaps()
        assert isinstance(gaps, pd.DataFrame)
        assert len(gaps) > 0

    def test_has_required_columns(self):
        gaps = get_department_gaps()
        assert "department" in gaps.columns
        assert "avg_gap" in gaps.columns
        assert "employee_count" in gaps.columns
