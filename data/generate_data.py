import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(1, 3650, n)
temp = 60 + 0.005 * age + np.random.normal(0, 5, n)
vibration = 0.5 + 0.0002 * age + np.random.normal(0, 0.1, n)
pressure = 100 - 0.003 * age + np.random.normal(0, 3, n)
rpm = 3000 - 0.05 * age + np.random.normal(0, 50, n)
oil_level = 100 - 0.01 * age + np.random.normal(0, 2, n)

failure = (
    (temp > 85) |
    (vibration > 1.0) |
    (pressure < 85) |
    (oil_level < 70)
).astype(int)

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
print(f"Dataset created: {len(df)} records, {df['failure'].sum()} failures ({df['failure'].mean()*100:.1f}%)")
