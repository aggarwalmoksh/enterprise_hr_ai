import pandas as pd

from app.services.skill_gap_service import get_employee_gaps
from app.utils.config import TRAINING_CATALOG
from app.utils.logger import get_logger

logger = get_logger("services.recommendation_service")

# Map specific O*NET skill name patterns to generic catalog keys
_SKILL_CATEGORY_MAP = {
    "crm": "CRM",
    "erp": "ERP software",
    "hris": "Human resource information system (HRIS)",
    "human resource": "Human resource information system (HRIS)",
    "payroll": "Payroll software",
    "accounting": "Accounting software",
    "spreadsheet": "Spreadsheet software",
    "word process": "Word processing software",
    "presentation": "Presentation software",
    "database": "Database reporting software",
    "sql": "SQL",
    "python": "Python",
    "java": "Java",
    "linux": "Linux",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "cloud": "Cloud",
    "microsoft excel": "Spreadsheet software",
    "microsoft word": "Word processing software",
    "microsoft powerpoint": "Presentation software",
    "microsoft outlook": "Communication software",
    "microsoft access": "Database reporting software",
    "microsoft project": "Project management software",
    "sharepoint": "Collaboration software",
    "sap": "SAP",
    "oracle": "Oracle Database",
    "salesforce": "Salesforce",
    "quickbooks": "QuickBooks",
    "sage": "Sage 50 Accounting",
    "adobe photoshop": "Adobe Photoshop",
    "adobe illustrator": "Adobe Illustrator",
    "adobe": "Adobe Creative Suite",
    "google analytics": "Analytics software",
    "google": "Collaboration software",
    "git": "Git",
    "github": "Git",
    "html": "Web development software",
    "css": "Web development software",
    "javascript": "Web development software",
    "react": "Web development software",
    "angular": "Web development software",
    "node": "Web development software",
    "api": "REST APIs",
    "rest": "REST APIs",
    "network": "Networking software",
    "firewall": "Security software",
    "security": "Security software",
    "backup": "Backup software",
    "virtual": "Virtualization software",
    "vmware": "VMware",
    "cisco": "Cisco",
    "ticketing": "Ticketing software",
    "help desk": "Help desk software",
    "service desk": "Service desk software",
    "project manage": "Project management software",
    "gantt": "Gantt chart software",
    "agile": "Agile software",
    "scrum": "Scrum software",
    "workflow": "Workflow software",
    "automation": "Automation software",
    "report": "Reporting software",
    "dashboard": "Dashboard software",
    "analytics": "Analytics software",
    "data analysis": "Data Analysis",
    "machine learning": "Machine Learning",
    "artificial intelligence": "Artificial intelligence software",
    "big data": "Big data software",
    "data visual": "Data visualization software",
    "tableau": "Data visualization software",
    "power bi": "Data visualization software",
    "etl": "ETL software",
    "data ware": "Data warehousing software",
    "learning manage": "Learning management software",
    "lms": "Learning management software",
    "training": "Training software",
    "onboard": "Onboarding software",
    "recruit": "Recruiting software",
    "applicant tracking": "Applicant tracking software",
    "talent manage": "Talent management software",
    "performance manage": "Performance management software",
    "compensation": "Compensation software",
    "benefits": "Benefits administration software",
    "compliance": "Compliance software",
    "safety": "Safety software",
    "incident": "Incident reporting software",
    "audit": "Audit software",
    "risk manage": "Risk management software",
    "contract manage": "Contract management software",
    "document manage": "Document management software",
    "content manage": "Content management software",
    "knowledge base": "Knowledge base software",
    "video conferenc": "Video conferencing software",
    "email": "Email software",
    "calendar": "Scheduling software",
    "scheduling": "Scheduling software",
    "time and attend": "Time and attendance software",
    "fleet": "Fleet management software",
    "inventory": "Inventory management software",
    "supply chain": "Supply chain software",
    "logistics": "Logistics software",
    "warehouse": "Warehouse management software",
    "procurement": "Procurement software",
    "expense": "Expense management software",
    "travel": "Travel management software",
}


def _match_skill_to_catalog(skill: str) -> str | None:
    """Match a specific O*NET skill name to a generic catalog key."""
    skill_lower = skill.lower()

    # 1. Exact match
    if skill in TRAINING_CATALOG:
        return skill

    # 2. Category pattern match
    for pattern, catalog_key in _SKILL_CATEGORY_MAP.items():
        if pattern in skill_lower:
            if catalog_key in TRAINING_CATALOG:
                return catalog_key

    return None


def get_recommendations(missing_skills: list[str]) -> list[dict]:
    recommendations = []
    seen = set()
    for skill in missing_skills:
        matched_key = _match_skill_to_catalog(skill)
        if matched_key and matched_key not in seen:
            course = TRAINING_CATALOG[matched_key]
            recommendations.append({
                "missing_skill": skill,
                "recommended_course": course["course"],
                "type": course["type"],
                "duration": course["duration"],
            })
            seen.add(matched_key)
    return recommendations


def get_employee_recommendations() -> pd.DataFrame:
    emp_gaps = get_employee_gaps()
    all_recs = []

    for _, row in emp_gaps.iterrows():
        recs = get_recommendations(row["skill_gaps"])
        if recs:
            top = recs[0]
            all_recs.append({
                "employee_id": row["employee_id"],
                "department": row["department"],
                "gap_count": row["gap_count"],
                "top_recommendation": top["recommended_course"],
                "training_duration": top["duration"],
                "training_type": top["type"],
                "all_recommendations": recs,
            })

    return pd.DataFrame(all_recs)


def get_training_priorities() -> pd.DataFrame:
    recs = get_employee_recommendations()
    if recs.empty:
        return pd.DataFrame()

    course_counts = recs["top_recommendation"].value_counts().reset_index()
    course_counts.columns = ["course", "employees_needing"]
    return course_counts
