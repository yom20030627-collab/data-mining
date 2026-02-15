import numpy as np
import pandas as pd

# ============================================================
# Upgraded, more realistic synthetic hospital + spine-surgery VTE dataset
# - Adds surgical variables & VTE outcomes inspired by a spine-surgery VTE meta-analysis:
#   age, female, diabetes, CKD, nonambulatory, D-dimer, long operation duration,
#   spine fusion, blood transfusion (as risk factors)  (see PMID/PMC meta-analysis)
# ============================================================

def logistic(x):
    return 1 / (1 + np.exp(-x))

def clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

np.random.seed(103)
N = 50000

# -------------------------
# 0) Demographics
# -------------------------
age = np.random.normal(55, 15, N)
age = clip(age, 18, 90).round().astype(int)

gender = np.random.choice(["Male", "Female"], size=N, p=[0.5, 0.5])
female = (gender == "Female").astype(int)

# Add race/ethnicity (optional, can remove)
race = np.random.choice(
    ["East_Asian", "White", "Black", "Hispanic", "Other"],
    size=N,
    p=[0.78, 0.08, 0.02, 0.07, 0.05]
)

# BMI with heavier tail
BMI = np.random.normal(25, 4.2, N) + np.random.gamma(shape=1.8, scale=0.5, size=N) - 0.9
BMI = clip(BMI, 16, 45)

smoking = np.random.choice([0, 1], N, p=[0.62, 0.38])
alcohol = np.random.choice([0, 1], N, p=[0.65, 0.35])

# Hormone therapy / OCP (only for females; simplistic)
hormone_tx = np.zeros(N, dtype=int)
f_idx = np.where(female == 1)[0]
hormone_tx[f_idx] = np.random.binomial(1, p=clip(0.10 - 0.0008*(age[f_idx]-35), 0.02, 0.12))

# Pregnancy (rare; only females age 18-45)
pregnancy = np.zeros(N, dtype=int)
pf = np.where((female == 1) & (age >= 18) & (age <= 45))[0]
pregnancy[pf] = np.random.binomial(1, p=0.03)

# -------------------------
# 1) Latent health factors (to create realistic correlation)
# -------------------------
# Metabolic burden: drives HTN/DM/lipids
metabolic = np.random.normal(0, 1, N) + 0.06*(BMI - 25) + 0.015*(age - 55)
# Inflammation burden: drives CRP, WBC
inflammation = np.random.normal(0, 1, N) + 0.25*smoking + 0.15*metabolic
# Hypercoagulability propensity: drives D-dimer, fibrinogen, VTE risk
hypercoag = np.random.normal(0, 1, N) + 0.25*inflammation + 0.10*(BMI-25)/5 + 0.15*hormone_tx + 0.35*pregnancy

# -------------------------
# 2) Comorbidities (probabilistic, not hard threshold)
# -------------------------
p_htn = logistic(-2.5 + 0.045*(age-45) + 0.10*(BMI-25) + 0.55*metabolic)
hypertension = np.random.binomial(1, clip(p_htn, 0.01, 0.95))

p_dm = logistic(-3.0 + 0.040*(age-45) + 0.12*(BMI-25) + 0.70*metabolic)
diabetes = np.random.binomial(1, clip(p_dm, 0.005, 0.85))

p_hld = logistic(-2.2 + 0.030*(age-45) + 0.08*(BMI-25) + 0.65*metabolic)
hyperlipidemia = np.random.binomial(1, clip(p_hld, 0.01, 0.90))

# CKD depends on age/DM/HTN + random
p_ckd = logistic(-3.2 + 0.035*(age-50) + 0.75*diabetes + 0.55*hypertension + 0.15*inflammation)
CKD = np.random.binomial(1, clip(p_ckd, 0.003, 0.60))

# CAD depends on risk factors
p_cad = logistic(-3.0 + 0.040*(age-50) + 0.55*hypertension + 0.45*hyperlipidemia + 0.35*smoking + 0.20*diabetes)
CAD = np.random.binomial(1, clip(p_cad, 0.002, 0.70))

# Prior VTE (rare but higher if cancer/CKD/older) - cancer generated below, so placeholder now
# We'll create cancer first, then update history_vte

# Stroke history
p_stroke = logistic(-4.0 + 0.045*(age-55) + 0.60*hypertension + 0.40*CAD)
stroke_history = np.random.binomial(1, clip(p_stroke, 0.001, 0.35))

# Cancer status (baseline + age + smoking)
p_cancer = logistic(-4.2 + 0.035*(age-55) + 0.35*smoking + 0.10*inflammation)
cancer = np.random.binomial(1, clip(p_cancer, 0.002, 0.25))
cancer_type = np.where(
    cancer == 1,
    np.random.choice(["Lung", "GI", "Breast", "GU", "Heme", "Other"], size=N, p=[0.25,0.20,0.18,0.12,0.08,0.17]),
    "None"
)
chemo_preop = np.where(cancer == 1, np.random.binomial(1, 0.22, N), 0)
chemo_postop = np.where(cancer == 1, np.random.binomial(1, 0.18, N), 0)

# Now prior VTE
p_prior_vte = logistic(-5.0 + 0.030*(age-55) + 0.70*cancer + 0.55*CKD + 0.25*hypercoag)
history_VTE = np.random.binomial(1, clip(p_prior_vte, 0.0005, 0.12))

# Thrombophilia (inherited/acquired; rare)
thrombophilia = np.random.binomial(1, clip(0.01 + 0.02*history_VTE, 0.005, 0.07))

# Varicose veins (mild risk factor; commoner with age/BMI)
p_varicose = logistic(-2.0 + 0.02*(age-50) + 0.06*(BMI-25))
varicose_veins = np.random.binomial(1, clip(p_varicose, 0.02, 0.45))

# Heart failure / COPD (optional but realistic)
p_chf = logistic(-4.0 + 0.04*(age-60) + 0.55*CAD + 0.35*CKD)
CHF = np.random.binomial(1, clip(p_chf, 0.001, 0.30))

p_copd = logistic(-4.1 + 0.03*(age-55) + 0.65*smoking)
COPD = np.random.binomial(1, clip(p_copd, 0.001, 0.25))

# -------------------------
# 3) Vitals (measurement noise + condition effects)
# -------------------------
systolic_bp = np.random.normal(118 + 14*hypertension + 3*CKD, 11, N)
diastolic_bp = np.random.normal(78 + 9*hypertension + 2*CKD, 8.5, N)
heart_rate = np.random.normal(74 + 4*CAD + 6*inflammation, 9, N)
resp_rate = np.random.normal(18 + 0.8*COPD, 2.2, N)
temperature = np.random.normal(36.8 + 0.15*inflammation, 0.35, N)
oxygen_sat = np.random.normal(98 - 2.0*COPD - 0.6*CAD, 1.2, N)

# -------------------------
# 4) Labs (with realism: skew + plausible ranges)
# -------------------------
# Glucose & HbA1c
fasting_glucose = np.random.normal(92 + 42*diabetes + 6*metabolic, 16, N)
HbA1c = np.random.normal(5.4 + 2.0*diabetes + 0.25*metabolic, 0.75, N)
HbA1c = clip(HbA1c, 4.5, 14.0)

# Lipids
total_chol = np.random.normal(178 + 45*hyperlipidemia + 6*metabolic, 32, N)
LDL = np.random.normal(98 + 38*hyperlipidemia + 4*metabolic, 26, N)
HDL = np.random.normal(52 - 9*hyperlipidemia - 2.5*metabolic, 11, N)
triglyceride = np.random.normal(120 + 75*hyperlipidemia + 10*metabolic, 45, N)

# Renal
creatinine = np.random.normal(0.95 + 0.85*CKD + 0.12*(age-55)/20, 0.25, N)
creatinine = clip(creatinine, 0.4, 8.0)
eGFR = 120 - age - 18*CKD + np.random.normal(0, 10, N)
eGFR = clip(eGFR, 3, 140)

# CBC + inflammation
WBC = np.random.normal(6.8 + 1.2*inflammation + 0.8*smoking, 1.8, N)
WBC = clip(WBC, 2.0, 25.0)
hemoglobin = np.random.normal(13.5 - 0.8*CKD - 0.2*inflammation - 0.6*female, 1.4, N)
hemoglobin = clip(hemoglobin, 6.5, 18.5)
platelet = np.random.normal(250 + 35*inflammation - 20*CKD, 55, N)
platelet = clip(platelet, 40, 900)

CRP = np.random.lognormal(mean=0.0 + 0.35*inflammation, sigma=0.7, size=N)  # mg/L-ish skewed
CRP = clip(CRP, 0.1, 250)

# Coagulation / hypercoag
# D-dimer in mg/L FEU-ish (skewed)
d_dimer = np.random.lognormal(mean=-0.3 + 0.45*hypercoag + 0.25*(age-55)/20 + 0.35*cancer, sigma=0.75, size=N)
d_dimer = clip(d_dimer, 0.05, 20.0)

fibrinogen = np.random.normal(3.0 + 0.6*hypercoag + 0.3*inflammation, 0.6, N)  # g/L-ish
fibrinogen = clip(fibrinogen, 1.0, 8.0)

INR = np.random.normal(1.0 + 0.03*CKD + 0.02*cancer, 0.08, N)
INR = clip(INR, 0.8, 2.5)

aPTT = np.random.normal(30 + 1.5*CKD + 0.8*cancer, 4.0, N)
aPTT = clip(aPTT, 18, 90)

# -------------------------
# 5) Surgery & peri-op variables (spine-surgery flavored but usable generally)
# -------------------------
surgery_type = np.random.choice(
    ["Spine_Fusion", "Spine_Decompression", "Spine_Tumor", "Other_Ortho", "General_Surgery"],
    size=N,
    p=[0.28, 0.30, 0.07, 0.20, 0.15]
)

spine_fusion = (surgery_type == "Spine_Fusion").astype(int)

# ASA score depends on age/comorbidity
asa_latent = -1.5 + 0.02*(age-55) + 0.55*CHF + 0.35*COPD + 0.25*CKD + 0.20*CAD + 0.25*cancer
asa_prob = logistic(asa_latent)
# Convert to ASA 1-4 (simple mapping)
ASA_score = np.where(asa_prob < 0.20, 1,
             np.where(asa_prob < 0.55, 2,
             np.where(asa_prob < 0.83, 3, 4)))

anesthesia_type = np.random.choice(["General", "Regional", "Combined"], size=N, p=[0.82, 0.12, 0.06])

# Operation duration (minutes): fusion/tumor longer
base_dur = np.random.normal(120, 35, N)
op_duration = base_dur + 55*spine_fusion + 70*(surgery_type=="Spine_Tumor").astype(int) + 18*(ASA_score>=3).astype(int)
op_duration = clip(op_duration, 30, 600).round().astype(int)

# Estimated blood loss (mL) & transfusion
blood_loss = np.random.lognormal(mean=5.2 + 0.25*spine_fusion + 0.35*(surgery_type=="Spine_Tumor").astype(int), sigma=0.6, size=N)
blood_loss = clip(blood_loss, 50, 4000)

p_transfusion = logistic(-5.0 + 0.0012*(blood_loss-300) + 0.35*(op_duration-120)/60 + 0.25*(hemoglobin<11).astype(int))
blood_transfusion = np.random.binomial(1, clip(p_transfusion, 0.001, 0.65))

# Preop ambulatory status (nonambulatory is a key risk factor)
p_nonambulatory = logistic(-3.0 + 0.03*(age-60) + 0.65*stroke_history + 0.45*CHF + 0.25*COPD)
nonambulatory_preop = np.random.binomial(1, clip(p_nonambulatory, 0.005, 0.35))

# Postop immobilization days
immob_days = np.random.poisson(lam=2 + 2.5*nonambulatory_preop + 0.9*(ASA_score>=3).astype(int) + 0.6*spine_fusion)
immob_days = clip(immob_days, 0, 30).astype(int)

# Central venous catheter (CVC)
p_cvc = logistic(-4.0 + 0.60*(ASA_score>=3).astype(int) + 0.35*cancer + 0.25*(op_duration>180).astype(int))
CVC = np.random.binomial(1, clip(p_cvc, 0.002, 0.35))

# Prophylaxis (pharmacologic + mechanical)
# In reality, prophylaxis depends on perceived risk and bleeding risk; we simulate both
bleed_risk = logistic(-2.2 + 0.40*(blood_loss>800).astype(int) + 0.35*(INR>1.3).astype(int) + 0.25*(platelet<120).astype(int))
vte_risk_screen = logistic(
    -3.4
    + 0.03*(age-60)
    + 0.65*history_VTE
    + 0.35*cancer
    + 0.35*nonambulatory_preop
    + 0.20*(d_dimer>0.8).astype(int)
    + 0.25*spine_fusion
)

# Pharmacologic prophylaxis less likely if bleed risk high
p_pharm_proph = clip(0.10 + 0.55*vte_risk_screen - 0.35*bleed_risk, 0.02, 0.85)
pharm_prophylaxis = np.random.binomial(1, p_pharm_proph)

# Mechanical prophylaxis more common and less constrained by bleed risk
p_mech_proph = clip(0.25 + 0.45*vte_risk_screen, 0.05, 0.95)
mech_prophylaxis = np.random.binomial(1, p_mech_proph)

# Chronic anticoagulant use preop (e.g., AF) - simplistic
p_chronic_ac = logistic(-4.2 + 0.05*(age-65) + 0.40*CAD + 0.35*CHF)
chronic_anticoag = np.random.binomial(1, clip(p_chronic_ac, 0.002, 0.30))

# Stop anticoag around surgery? (bridge)
bridge_therapy = np.where(chronic_anticoag==1, np.random.binomial(1, 0.25, N), 0)

# -------------------------
# 6) Outcomes: ICU, Mortality, LOS + VTE (DVT/PE)
# -------------------------
# ICU admission (more realistic, depends on surgery, ASA, comorbidity)
p_icu = logistic(
    -4.0
    + 0.55*(ASA_score>=3).astype(int)
    + 0.35*CHF
    + 0.25*CKD
    + 0.25*(op_duration>180).astype(int)
    + 0.25*(blood_loss>800).astype(int)
    + 0.20*(surgery_type=="Spine_Tumor").astype(int)
)
ICU_admission = np.random.binomial(1, clip(p_icu, 0.01, 0.70))

# Length of stay: base + surgery + ICU + immob
length_of_stay = np.random.normal(4.5, 1.8, N) + 2.2*spine_fusion + 1.5*(surgery_type=="Spine_Tumor").astype(int) + 4.5*ICU_admission + 0.35*immob_days
length_of_stay = clip(length_of_stay, 1, 60).round().astype(int)

# VTE risk model (key: match meta-analysis directions + add realism)
# NOTE: We model VTE AFTER surgery; prophylaxis reduces risk.
# Base incidence tuned low (~0.3-0.6%) depending on risk distribution.
vte_logit = (
    -7.2
    + 0.030*(age-60)              # age
    + 0.18*female                 # female
    + 0.55*diabetes               # diabetes
    + 0.95*CKD                    # CKD (strong)
    + 0.90*nonambulatory_preop    # nonambulatory
    + 0.45*np.log1p(d_dimer)      # D-dimer
    + 0.30*(op_duration-120)/60   # duration
    + 0.35*spine_fusion           # fusion
    + 0.60*blood_transfusion      # transfusion
    # extra, clinically plausible
    + 0.75*history_VTE
    + 0.45*cancer
    + 0.35*thrombophilia
    + 0.25*CVC
    + 0.20*varicose_veins
    + 0.25*hormone_tx
    + 0.35*pregnancy
    + 0.15*(BMI>=30).astype(int)
)

# Prophylaxis effects (reduce risk)
vte_logit += -0.45*pharm_prophylaxis + -0.20*mech_prophylaxis

# Convert to probability; cap to keep rare-event nature
p_vte = clip(logistic(vte_logit), 0.0002, 0.08)

VTE = np.random.binomial(1, p_vte)

# Split into DVT vs PE
# PE more likely if higher hypercoag + longer immob + cancer
pe_logit = -1.8 + 0.35*hypercoag + 0.25*(immob_days>5).astype(int) + 0.30*cancer + 0.20*CHF
p_pe_given_vte = clip(logistic(pe_logit), 0.10, 0.55)
PE = np.where(VTE==1, np.random.binomial(1, p_pe_given_vte), 0)
DVT = np.where(VTE==1, 1-PE, 0)

# Time-to-event (days after surgery): most within first month; use Weibull-like
# If no event -> NaN
shape = 1.4
scale = 9.0  # controls typical event time
u = np.random.uniform(size=N)
tte = scale * (-np.log(1-u))**(1/shape)
# shift earlier for high-risk
tte = tte / (1 + 0.20*(immob_days>5).astype(int) + 0.15*cancer + 0.10*(d_dimer>1.0).astype(int))
VTE_time_to_event_days = np.where(VTE==1, clip(tte, 0.5, 60.0), np.nan)

# Mortality: depends on age, ICU, severe comorbidity, and PE (a bit)
mort_logit = (
    -6.2
    + 0.045*(age-65)
    + 1.25*ICU_admission
    + 0.45*CHF
    + 0.35*CKD
    + 0.25*cancer
    + 0.40*PE
)
mortality = np.random.binomial(1, clip(logistic(mort_logit), 0.001, 0.35))

# -------------------------
# 7) Introduce some missingness (real-world data)
# -------------------------
def apply_missing(x, missing_rate):
    x = x.astype("float") if np.issubdtype(x.dtype, np.number) else x.copy()
    m = np.random.binomial(1, missing_rate, size=len(x)).astype(bool)
    if x.dtype == "float" or x.dtype == "int" or np.issubdtype(x.dtype, np.number):
        x[m] = np.nan
    else:
        x[m] = None
    return x

# labs sometimes missing
HbA1c_m = apply_missing(HbA1c, 0.18)
CRP_m = apply_missing(CRP, 0.22)
aPTT_m = apply_missing(aPTT, 0.25)
INR_m = apply_missing(INR, 0.20)
fibrinogen_m = apply_missing(fibrinogen, 0.30)
d_dimer_m = apply_missing(d_dimer, 0.28)

# -------------------------
# 8) Build DataFrame
# -------------------------
df = pd.DataFrame({
    "patient_id": np.arange(1, N+1),

    # Demographics
    "age": age,
    "gender": gender,
    "race": race,
    "BMI": BMI.round(2),
    "smoking": smoking,
    "alcohol": alcohol,
    "hormone_therapy": hormone_tx,
    "pregnancy": pregnancy,

    # Comorbidities
    "hypertension": hypertension,
    "diabetes": diabetes,
    "hyperlipidemia": hyperlipidemia,
    "CKD": CKD,
    "CAD": CAD,
    "stroke_history": stroke_history,
    "CHF": CHF,
    "COPD": COPD,
    "cancer": cancer,
    "cancer_type": cancer_type,
    "chemo_preop": chemo_preop,
    "chemo_postop": chemo_postop,
    "history_VTE": history_VTE,
    "thrombophilia": thrombophilia,
    "varicose_veins": varicose_veins,

    # Vitals
    "systolic_bp": systolic_bp.round(1),
    "diastolic_bp": diastolic_bp.round(1),
    "heart_rate": heart_rate.round(1),
    "resp_rate": resp_rate.round(1),
    "temperature": temperature.round(2),
    "oxygen_sat": oxygen_sat.round(1),

    # Labs (with missingness versions)
    "fasting_glucose": fasting_glucose.round(1),
    "HbA1c": np.round(HbA1c_m, 2),
    "total_cholesterol": total_chol.round(1),
    "LDL": LDL.round(1),
    "HDL": HDL.round(1),
    "triglyceride": triglyceride.round(1),
    "creatinine": creatinine.round(2),
    "eGFR": eGFR.round(1),
    "WBC": WBC.round(2),
    "hemoglobin": hemoglobin.round(2),
    "platelet_count": platelet.round(0),
    "CRP": np.round(CRP_m, 2),

    "D_dimer": np.round(d_dimer_m, 3),
    "fibrinogen": np.round(fibrinogen_m, 2),
    "INR": np.round(INR_m, 2),
    "aPTT": np.round(aPTT_m, 1),

    # Surgery / peri-op
    "surgery_type": surgery_type,
    "spine_fusion": spine_fusion,
    "ASA_score": ASA_score,
    "anesthesia_type": anesthesia_type,
    "operation_duration_min": op_duration,
    "estimated_blood_loss_ml": blood_loss.round(0),
    "blood_transfusion": blood_transfusion,
    "nonambulatory_preop": nonambulatory_preop,
    "postop_immobilization_days": immob_days,
    "central_venous_catheter": CVC,
    "chronic_anticoagulation": chronic_anticoag,
    "bridge_therapy": bridge_therapy,
    "pharm_prophylaxis": pharm_prophylaxis,
    "mech_prophylaxis": mech_prophylaxis,

    # Hospital outcomes
    "ICU_admission": ICU_admission,
    "length_of_stay": length_of_stay,
    "mortality": mortality,

    # VTE outcomes
    "VTE": VTE,
    "DVT": DVT,
    "PE": PE,
    "VTE_time_to_event_days": np.round(VTE_time_to_event_days, 2),
})

# Save
out_path = "simulated_hospital_vte_upgraded_dataset.csv"
df.to_csv(out_path, index=False)

print("✅ 升級版大型醫院 + 手術 VTE 模擬資料已生成")
print("檔案:", out_path)
print("總筆數:", len(df))
print("VTE incidence:", df["VTE"].mean())
print("PE incidence:", df["PE"].mean())
print("DVT incidence:", df["DVT"].mean())
