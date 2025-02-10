import pandas as pd
import numpy as np

# Hasta verilerini yükleyin
data_path = r"C:\........\patient.csv"  # yada synthetic_data_generation'dan oluşturulmuş veri
df_patients = pd.read_csv(data_path)


def calculate_severity_and_priority(row):
    
    
    severity = np.random.uniform(0, 1)
    row['Severity'] = severity

    #öncelik skoru hesaplama (ciddiyet ve yaşa dayalı)
    priority = 0.7 * severity + 0.3 * (row['Age'] / 100)
    row['Priority'] = priority
    return row

df_patients = df_patients.apply(calculate_severity_and_priority, axis=1)


df_patients.to_csv("processed_patient_data.csv", index=False)
