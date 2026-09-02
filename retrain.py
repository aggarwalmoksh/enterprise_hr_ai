import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import json
import datetime
import os

# Ensure directories exist
os.makedirs("models/v2", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

df = pd.read_csv("data/raw/employee_attrition.csv")

# Process it
df.to_csv("data/processed/employee_attrition_processed.csv", index=False)

# Re-engineer features (like in predictor)
print("Engineering features...")
df_engineered = df.copy()
if "MonthlySalary" in df.columns:
    df_engineered["Income_per_year"] = df_engineered["MonthlySalary"] * 12
if "LastPromotionYear" in df.columns:
    df_engineered["Gap_since_promotion"] = datetime.datetime.now().year - df_engineered["LastPromotionYear"]
if "CustomerSatisfaction" in df.columns and "WorkLifeBalanceScore" in df.columns:
    df_engineered["Satisfaction_score"] = (df_engineered["CustomerSatisfaction"] * 0.5 + df_engineered["WorkLifeBalanceScore"] * 0.5)
if "YearsAtCompany" in df.columns and "Age" in df.columns:
    df_engineered["Experience_ratio"] = df_engineered["YearsAtCompany"] / df_engineered["Age"].clip(lower=1)

# Target
y = (df_engineered["AttritionRisk"] == "Yes").astype(int)

# Features
numeric_cols = ["Age", "MonthlySalary", "OvertimeHoursPerMonth", "LeavesTaken", "ProjectsHandled", "TrainingHours", 
                "CustomerSatisfaction", "YearsAtCompany", "WorkLifeBalanceScore", "PerformanceRating",
                "Income_per_year", "Gap_since_promotion", "Satisfaction_score", "Experience_ratio"]
categorical_cols = ["Gender", "Department", "JobRole", "EducationLevel", "Country"]

# Keep only those that exist
numeric_cols = [c for c in numeric_cols if c in df_engineered.columns]
categorical_cols = [c for c in categorical_cols if c in df_engineered.columns]

X = df_engineered[numeric_cols + categorical_cols]

X.to_csv("data/features/X_features.csv", index=False)
y.to_csv("data/features/y_target.csv", index=False)

metadata = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target": "AttritionRisk"
}
with open("data/features/column_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
])

pipeline.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score, f1_score
probs = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, probs)
preds = pipeline.predict(X_test)
f1 = f1_score(y_test, preds)

print(f"Model trained. ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}")

# Save
joblib.dump(pipeline, "models/v2/attrition_pipeline.joblib")
joblib.dump(pipeline, "models/attrition_pipeline.joblib") # overwrite latest

metadata_log = {
    "model_name": "Attrition Prediction Model",
    "version": "v2.0",
    "algorithm": "XGBoost",
    "training_date": str(datetime.datetime.now()),
    "roc_auc": float(roc_auc),
    "f1_score": float(f1)
}
with open("models/v2/metadata.json", "w") as f:
    json.dump(metadata_log, f, indent=4)

with open("models/version_log.json", "w") as f:
    json.dump(metadata_log, f, indent=4)

print("Retraining complete. Saved to models/v2/")
