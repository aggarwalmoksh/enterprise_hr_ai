import ast
import pandas as pd

from app.utils.config import (
    ROLE_MASTER_FILE,
    EMPLOYEE_SKILLS_FILE,
    EMPLOYEE_FILE,
    DEPT_KEYWORDS,
)
from app.utils.logger import get_logger

logger = get_logger("services.skill_gap_service")


def _safe_parse_list(val):
    if isinstance(val, str) and val.startswith("["):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return val if isinstance(val, list) else []


def _map_to_department(title: str) -> str:
    if pd.isna(title):
        return "Other"
    title_lower = str(title).lower()
    for dept, keywords in DEPT_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return dept
    return "Other"


def _build_dept_lookup() -> dict:
    role_master = pd.read_csv(ROLE_MASTER_FILE)
    role_master["Mapped_Department"] = role_master["Title"].apply(_map_to_department)

    dept_required = role_master.groupby("Mapped_Department").agg(
        Essential_Skills=("Essential_Skills", lambda x: set(
            item for sublist in x for item in _safe_parse_list(sublist)
        )),
        Software_Skills=("Software_Skills", lambda x: set(
            item for sublist in x for item in _safe_parse_list(sublist)
        )),
    ).reset_index()

    dept_required["All_Required"] = dept_required.apply(
        lambda row: row["Essential_Skills"].union(row["Software_Skills"]), axis=1
    )
    return dict(zip(dept_required["Mapped_Department"], dept_required["All_Required"]))


def get_employee_gaps() -> pd.DataFrame:
    emp = pd.read_csv(EMPLOYEE_FILE)
    emp_skills = pd.read_csv(EMPLOYEE_SKILLS_FILE)
    dept_lookup = _build_dept_lookup()

    skill_sets = emp_skills.groupby("EmployeeID")["current_skill"].apply(set).reset_index()
    skill_sets.columns = ["EmployeeID", "Current_Skills"]
    skill_sets = skill_sets.merge(emp[["EmployeeID", "Department"]], on="EmployeeID")

    gaps = []
    for _, row in skill_sets.iterrows():
        required = dept_lookup.get(row["Department"], set())
        gap = required - row["Current_Skills"]
        gaps.append({
            "employee_id": row["EmployeeID"],
            "department": row["Department"],
            "gap_count": len(gap),
            "skill_gaps": sorted(list(gap))[:10],
        })

    return pd.DataFrame(gaps)


def get_org_wide_gaps() -> pd.DataFrame:
    from collections import Counter
    emp_gaps = get_employee_gaps()
    all_gaps = []
    for gaps_list in emp_gaps["skill_gaps"]:
        all_gaps.extend(gaps_list)

    counter = Counter(all_gaps)
    result = pd.DataFrame(counter.most_common(), columns=["skill", "employees_missing"])

    def severity(count):
        if count >= 100:
            return "HIGH"
        elif count >= 50:
            return "MEDIUM"
        return "LOW"

    result["severity"] = result["employees_missing"].apply(severity)
    return result


def get_department_gaps() -> pd.DataFrame:
    emp_gaps = get_employee_gaps()
    dept_summary = emp_gaps.groupby("department").agg(
        avg_gap=("gap_count", "mean"),
        employee_count=("employee_id", "count"),
    ).round(1).sort_values("avg_gap", ascending=False).reset_index()
    return dept_summary
