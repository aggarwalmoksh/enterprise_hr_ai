import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.ml.model_loader import load_pipeline, load_metadata
from app.ml.predictor import predict_attrition, predict_single
from app.services import skill_gap_service, recommendation_service
from app.utils.config import EMPLOYEE_FILE, ENGAGEMENT_FILE

st.set_page_config(
    page_title="AI Workforce Intelligence Platform",
    page_icon="🧠",
    layout="wide",
)

@st.cache_data
def load_data():
    emp = pd.read_csv(EMPLOYEE_FILE)
    eng = pd.read_csv(ENGAGEMENT_FILE)
    return emp, eng

@st.cache_data
def get_predictions(_emp):
    return predict_attrition(_emp)

@st.cache_data
def get_skill_gaps():
    return skill_gap_service.get_employee_gaps()

@st.cache_data
def get_org_gaps():
    return skill_gap_service.get_org_wide_gaps()

@st.cache_data
def get_recommendations():
    return recommendation_service.get_employee_recommendations()


def main():
    load_pipeline()
    load_metadata()

    emp, eng = load_data()
    preds = get_predictions(emp)
    emp_gaps = get_skill_gaps()
    org_gaps = get_org_gaps()
    recs = get_recommendations()

    merged = emp[["EmployeeID", "Department", "JobRole", "Age", "Gender"]].merge(preds, on="EmployeeID")

    st.title("🧠 AI WORKFORCE INTELLIGENCE PLATFORM")
    st.divider()

    # --- KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Employees", f"{len(emp):,}")
    with col2:
        high_risk = int((preds["risk"] == "HIGH").sum())
        st.metric("High Risk", f"{high_risk:,}", delta=f"{high_risk/len(emp)*100:.1f}%")
    with col3:
        avg_eng = round(float(eng["Engagement Score"].mean()), 1)
        st.metric("Avg Engagement", f"{avg_eng}%")
    with col4:
        avg_prob = round(float(preds["attrition_probability"].mean()) * 100, 1)
        st.metric("Avg Attrition Prob", f"{avg_prob}%")

    st.divider()

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    departments = ["All"] + sorted(merged["Department"].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)

    if selected_dept != "All":
        filtered = merged[merged["Department"] == selected_dept]
        filtered_gaps = emp_gaps[emp_gaps["department"] == selected_dept]
    else:
        filtered = merged
        filtered_gaps = emp_gaps

    # --- ROW 1: CHARTS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Attrition Risk by Department")
        dept_risk = merged.groupby("Department").agg(
            total=("EmployeeID", "count"),
            high_risk=("risk", lambda x: (x == "HIGH").sum()),
            avg_prob=("attrition_probability", "mean"),
        ).reset_index()

        fig_dept = px.bar(
            dept_risk,
            x="Department",
            y=["high_risk", "total"],
            barmode="overlay",
            color_discrete_map={"high_risk": "#d32f2f", "total": "#90caf9"},
            labels={"value": "Count", "variable": ""},
        )
        fig_dept.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig_dept, use_container_width=True)

    with col_right:
        st.subheader("🎯 Risk Distribution")
        risk_counts = filtered["risk"].value_counts()
        colors = {"HIGH": "#d32f2f", "MEDIUM": "#ff9800", "LOW": "#4caf50"}
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=colors,
            hole=0.4,
        )
        fig_risk.update_layout(height=350, margin=dict(t=30))
        st.plotly_chart(fig_risk, use_container_width=True)

    # --- ROW 2: SKILL GAPS ---
    st.divider()
    st.subheader("🔴 Critical Organisation Skill Gaps")

    critical = org_gaps[org_gaps["severity"].isin(["HIGH", "MEDIUM"])].head(15)
    severity_colors = {"HIGH": "#d32f2f", "MEDIUM": "#ff9800", "LOW": "#4caf50"}

    fig_gaps = px.bar(
        critical,
        x="employees_missing",
        y="skill",
        color="severity",
        color_discrete_map=severity_colors,
        orientation="h",
        labels={"employees_missing": "Employees Missing", "skill": ""},
    )
    fig_gaps.update_layout(height=400, margin=dict(t=30), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_gaps, use_container_width=True)

    # --- ROW 3: RECOMMENDATIONS ---
    st.divider()
    st.subheader("📚 AI Upskilling Recommendations")

    if not recs.empty:
        top_recs = recs.groupby("top_recommendation").agg(
            employees=("employee_id", "count"),
        ).sort_values("employees", ascending=False).head(10).reset_index()

        col_rec1, col_rec2 = st.columns([1, 2])

        with col_rec1:
            st.dataframe(
                top_recs.rename(columns={"top_recommendation": "Course", "employees": "Employees"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_rec2:
            fig_recs = px.bar(
                top_recs,
                x="employees",
                y="top_recommendation",
                orientation="h",
                color="employees",
                color_continuous_scale="Teal",
                labels={"employees": "Employees", "top_recommendation": ""},
            )
            fig_recs.update_layout(height=350, margin=dict(t=30), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_recs, use_container_width=True)
    else:
        st.info("No recommendations available")

    # --- ROW 4: EMPLOYEE DRILL-DOWN ---
    st.divider()
    st.subheader("🔍 Employee Drill-Down")

    col_search1, col_search2 = st.columns([1, 3])
    with col_search1:
        emp_id = st.number_input("Employee ID", min_value=1, max_value=int(emp["EmployeeID"].max()), value=1)

    with col_search2:
        if st.button("Look Up Employee", type="primary"):
            emp_row = emp[emp["EmployeeID"] == emp_id]
            if emp_row.empty:
                st.error(f"Employee {emp_id} not found")
            else:
                emp_data = emp_row.iloc[0]
                pred = predict_single(emp_data.to_dict())
                emp_gap = emp_gaps[emp_gaps["employee_id"] == emp_id]
                gaps = emp_gap.iloc[0]["skill_gaps"] if not emp_gap.empty else []
                emp_recs = recommendation_service.get_recommendations(gaps)

                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Department", emp_data["Department"])
                with c2:
                    st.metric("Role", emp_data["JobRole"])
                with c3:
                    risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                    st.metric("Risk", f"{risk_color[pred['risk']]} {pred['risk']}")
                with c4:
                    st.metric("Attrition Prob", f"{pred['attrition_probability']:.1%}")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Top Skill Gaps:**")
                    if gaps:
                        for g in gaps[:5]:
                            st.write(f"- {g}")
                    else:
                        st.write("No gaps identified")

                with col_b:
                    st.write("**Recommendations:**")
                    if emp_recs:
                        for r in emp_recs[:3]:
                            st.write(f"- **{r['missing_skill']}** → {r['recommended_course']} ({r['duration']})")
                    else:
                        st.write("No recommendations")


if __name__ == "__main__":
    main()
