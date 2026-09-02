import pandas as pd
from datetime import datetime
import os
import json

def monitor_drift():
    print('Starting Data Drift Monitoring...')
    
    # Load training data
    train_data_path = 'data/processed/employee_attrition_processed.csv'
    if not os.path.exists(train_data_path):
        print('Training data not found. Run model retraining first.')
        return
        
    train_df = pd.read_csv(train_data_path)
    
    # Normally we would load production prediction logs here.
    # For demonstration, we simulate production logs.
    print(f'Training data size: {len(train_df)}')
    
    # Basic statistics comparison
    metrics_to_watch = ['Age', 'MonthlySalary', 'YearsAtCompany', 'JobSatisfaction']
    
    drift_detected = False
    drift_report = {}
    
    for metric in metrics_to_watch:
        if metric in train_df.columns:
            train_mean = train_df[metric].mean()
            # Simulate prod data slightly drifted
            prod_mean = train_mean * 1.05 
            
            drift_percent = abs(prod_mean - train_mean) / train_mean * 100
            drift_report[metric] = {
                'train_mean': float(train_mean),
                'prod_mean': float(prod_mean),
                'drift_percentage': float(drift_percent)
            }
            
            if drift_percent > 10:  # 10% threshold
                drift_detected = True
                
    with open('models/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=4)
        
    if drift_detected:
        print('WARNING: Data drift detected! Consider retraining the model.')
        # In a real system, this would trigger an alert or a retraining pipeline
    else:
        print('Data is stable. No significant drift detected.')

if __name__ == '__main__':
    monitor_drift()
