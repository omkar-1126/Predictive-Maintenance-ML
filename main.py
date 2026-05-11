import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import *
from src.preprocess import load_and_preprocess
from src.model import train_and_evaluate
from src.visualize import (
    plot_eda, plot_correlation, plot_confusion_matrices,
    plot_roc_curves, plot_feature_importance, plot_dashboard
)

if __name__ == '__main__':
    print("Step 1: Generating sensor dataset...")
    os.makedirs('data', exist_ok=True)

    import numpy as np
    np.random.seed(42)
    n = 2000
    age = np.random.randint(1, 3650, n)
    temp = 60 + 0.005 * age + np.random.normal(0, 5, n)
    vibration = 0.5 + 0.0002 * age + np.random.normal(0, 0.1, n)
    pressure = 100 - 0.003 * age + np.random.normal(0, 3, n)
    rpm = 3000 - 0.05 * age + np.random.normal(0, 50, n)
    oil_level = 100 - 0.01 * age + np.random.normal(0, 2, n)
    failure = ((temp > 85) | (vibration > 1.0) | (pressure < 85) | (oil_level < 70)).astype(int)

    df = pd.DataFrame({
        'equipment_age_days': age,
        'temperature_C': np.round(temp, 2),
        'vibration_mm_s': np.round(vibration, 3),
        'pressure_bar': np.round(pressure, 2),
        'rpm': np.round(rpm, 1),
        'oil_level_pct': np.round(np.clip(oil_level, 0, 100), 2),
        'failure': failure
    })
    df.to_csv('data/sensor_data.csv', index=False)
    print(f"Dataset: {len(df)} records | Failures: {df['failure'].sum()} ({df['failure'].mean()*100:.1f}%)")

    print("\nStep 2: Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess('data/sensor_data.csv')

    print("\nStep 3: Training & Evaluating Models...")
    results, best_name, importance_df = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    print("\nStep 4: Generating Visualizations...")
    plot_eda(df)
    plot_correlation(df)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_feature_importance(importance_df)
    plot_dashboard(df, results, importance_df, y_test)

    print("\nAll done! Check the outputs/ folder for charts.")
    print(f"Best Model: {best_name} | ROC-AUC: {results[best_name]['roc_auc']:.4f}")
