import pandas as pd
import numpy as np

np.random.seed(103)  # 固定隨機種子以確保可重現性

N = 50000  # 病患數量

# =========================
# 基本資料
# =========================
age = np.random.normal(55, 15, N).clip(18, 90).astype(int)
gender = np.random.choice(['Male', 'Female'], N)
BMI = np.random.normal(25, 4, N).clip(16, 40)
smoking = np.random.choice([0, 1], N, p=[0.6, 0.4])
alcohol = np.random.choice([0, 1], N, p=[0.6, 0.4])

# =========================
# 慢性病機率模型
# =========================
hypertension = (0.03*age + 0.04*BMI + np.random.normal(0,2,N)) > 10
diabetes = (0.03*age + 0.08*BMI + np.random.normal(0,2,N)) > 9
hyperlipidemia = (0.02*age + 0.04*BMI + np.random.normal(0,2,N)) > 9
CKD = (diabetes*3 + hypertension*1.5 + np.random.normal(0,2,N)) > 3
CAD = (hypertension*2 + hyperlipidemia*2 + smoking*1.5 + np.random.normal(0,2,N)) > 4
stroke = (hypertension*2.5 + CAD*2.5 + np.random.normal(0,2,N)) > 4

# =========================
# 生理數據
# =========================
systolic_bp = np.random.normal(120 + hypertension*15, 10)
diastolic_bp = np.random.normal(80 + hypertension*10, 8)
heart_rate = np.random.normal(75 + CAD*5, 8)
resp_rate = np.random.normal(18, 2)
temperature = np.random.normal(36.8, 0.3)
oxygen = np.random.normal(98 - CAD*2, 1)

# =========================
# 實驗室
# =========================
fasting_glucose = np.random.normal(90 + diabetes*40, 15)
HbA1c = np.random.normal(5.5 + diabetes*2, 0.8)
total_chol = np.random.normal(180 + hyperlipidemia*50, 30)
LDL = np.random.normal(100 + hyperlipidemia*40, 25)
HDL = np.random.normal(50 - hyperlipidemia*10, 10)
triglyceride = np.random.normal(120 + hyperlipidemia*80, 40)
creatinine = np.random.normal(1 + CKD*1.2, 0.3)
eGFR = 120 - age - CKD*20 + np.random.normal(0,10,N)

# =========================
# 住院與預後
# =========================
ICU = (CAD*2 + CKD*2 + stroke*3 + np.random.normal(0,2,N)) > 4
mortality = (ICU*3 + age*0.05 + np.random.normal(0,2,N)) > 6
length_of_stay = np.random.normal(5 + ICU*7, 2).clip(1,30)

# =========================
# 組合成 DataFrame
# =========================
df = pd.DataFrame({
    "patient_id": range(1,N+1),
    "age": age,
    "gender": gender,
    "BMI": BMI,
    "smoking": smoking,
    "alcohol": alcohol,
    "hypertension": hypertension.astype(int),
    "diabetes": diabetes.astype(int),
    "hyperlipidemia": hyperlipidemia.astype(int),
    "CKD": CKD.astype(int),
    "CAD": CAD.astype(int),
    "stroke_history": stroke.astype(int),
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
    "resp_rate": resp_rate,
    "temperature": temperature,
    "oxygen_sat": oxygen,
    "fasting_glucose": fasting_glucose,
    "HbA1c": HbA1c,
    "total_cholesterol": total_chol,
    "LDL": LDL,
    "HDL": HDL,
    "triglyceride": triglyceride,
    "creatinine": creatinine,
    "eGFR": eGFR,
    "ICU_admission": ICU.astype(int),
    "mortality": mortality.astype(int),
    "length_of_stay": length_of_stay
})

df.to_csv("simulated_hospital_large_dataset.csv", index=False)

print("✅ 大型醫院模擬資料已生成")
print("總筆數:", len(df))

