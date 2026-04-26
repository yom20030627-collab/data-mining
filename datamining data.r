import pandas as pd
import numpy as np

# =========================
# 1. Read original metadata
# =========================
df = pd.read_csv("weis_metadata_1210.csv")

np.random.seed(42)
n = len(df)

# =========================
# 2. Helper functions
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normal_clip(mean, sd, size=n, low=0, high=1):
    return np.clip(np.random.normal(mean, sd, size), low, high)

# =========================
# 3. Standardize key columns
# =========================

# Try to detect entacapone column
possible_ent_cols = [c for c in df.columns if "entacapone" in c.lower()]
if possible_ent_cols:
    ent_col = possible_ent_cols[0]
    df["Entacapone"] = df[ent_col].astype(str).str.lower().isin(
        ["yes", "y", "1", "true", "pd_le", "le"]
    ).astype(int)
else:
    df["Entacapone"] = np.random.binomial(1, 11/24, n)

# Try to detect sex column
possible_sex_cols = [c for c in df.columns if c.lower() in ["sex", "gender"]]
if possible_sex_cols:
    sex_col = possible_sex_cols[0]
    df["Sex_binary"] = df[sex_col].astype(str).str.lower().map({
        "male": 1, "m": 1,
        "female": 0, "f": 0
    })
else:
    df["Sex_binary"] = np.random.binomial(1, 0.6, n)

df["Sex_binary"] = df["Sex_binary"].fillna(np.random.binomial(1, 0.6, n))

# =========================
# 4. Microbiome proxy features
# =========================

df["Faecalibacterium"] = normal_clip(0.45, 0.12)
df["Lactobacillus"] = normal_clip(0.35, 0.10)
df["Bifidobacterium"] = normal_clip(0.25, 0.10)
df["Dorea"] = normal_clip(0.30, 0.10)
df["Blautia"] = normal_clip(0.32, 0.10)

df["Methanobrevibacter"] = normal_clip(0.18, 0.08)
df["Intestinimonas"] = normal_clip(0.20, 0.08)

# Entacapone-associated effects
ent = df["Entacapone"] == 1

df.loc[ent, "Faecalibacterium"] *= np.random.normal(0.55, 0.08, ent.sum())
df.loc[ent, "Lactobacillus"] *= np.random.normal(0.60, 0.08, ent.sum())
df.loc[ent, "Bifidobacterium"] *= np.random.normal(1.80, 0.20, ent.sum())

df["Faecalibacterium"] = df["Faecalibacterium"].clip(0, 1)
df["Lactobacillus"] = df["Lactobacillus"].clip(0, 1)
df["Bifidobacterium"] = df["Bifidobacterium"].clip(0, 1)

# =========================
# 5. Functional pathway proxy features
# =========================

df["Nicotinate_metabolism"] = normal_clip(0.55, 0.12)
df["Caffeine_metabolism"] = normal_clip(0.50, 0.12)
df["Xenobiotic_degradation"] = normal_clip(0.48, 0.12)
df["Indole_metabolism"] = normal_clip(0.45, 0.12)
df["Amino_acid_metabolism"] = normal_clip(0.40, 0.12)

df.loc[ent, "Amino_acid_metabolism"] *= np.random.normal(1.45, 0.15, ent.sum())
df.loc[ent, "Xenobiotic_degradation"] *= np.random.normal(1.20, 0.12, ent.sum())

for col in [
    "Nicotinate_metabolism",
    "Caffeine_metabolism",
    "Xenobiotic_degradation",
    "Indole_metabolism",
    "Amino_acid_metabolism"
]:
    df[col] = df[col].clip(0, 1)

# =========================
# 6. Generate ICD label
# =========================

icd_score = (
    -3.2
    + 3.0 * df["Methanobrevibacter"]
    + 2.4 * df["Intestinimonas"]
    - 1.5 * df["Nicotinate_metabolism"]
    - 1.0 * df["Caffeine_metabolism"]
    + 0.4 * df["Sex_binary"]
    + np.random.normal(0, 0.35, n)
)

icd_prob = sigmoid(icd_score)
target_icd_rate = 0.08
target_icd_n = max(1, int(round(n * target_icd_rate)))

df["ICD_risk_score"] = icd_prob
top_icd_idx = df["ICD_risk_score"].nlargest(target_icd_n).index
df["ICD"] = 0
df.loc[top_icd_idx, "ICD"] = 1

# ICD-associated microbiome adjustment
icd = df["ICD"] == 1

df.loc[icd, "Methanobrevibacter"] *= np.random.normal(1.8, 0.15, icd.sum())
df.loc[icd, "Intestinimonas"] *= np.random.normal(1.6, 0.15, icd.sum())
df.loc[icd, "Nicotinate_metabolism"] *= np.random.normal(0.65, 0.08, icd.sum())
df.loc[icd, "Caffeine_metabolism"] *= np.random.normal(0.70, 0.08, icd.sum())

for col in [
    "Methanobrevibacter",
    "Intestinimonas",
    "Nicotinate_metabolism",
    "Caffeine_metabolism"
]:
    df[col] = df[col].clip(0, 1)

# =========================
# 7. Generate GI side effect label
# =========================

gi_score = (
    -1.6
    + 1.5 * df["Entacapone"]
    - 1.2 * df["Faecalibacterium"]
    - 0.8 * df["Lactobacillus"]
    + 0.8 * df["Bifidobacterium"]
    + 1.0 * df["Amino_acid_metabolism"]
    + np.random.normal(0, 0.35, n)
)

df["GI_side_effect_risk_score"] = sigmoid(gi_score)
df["GI_side_effect"] = np.random.binomial(1, df["GI_side_effect_risk_score"])

# =========================
# 8. Engineered features
# =========================

df["Microbiome_dysbiosis_score"] = (
    (1 - df["Faecalibacterium"])
    + (1 - df["Lactobacillus"])
    + df["Bifidobacterium"]
    + df["Methanobrevibacter"]
) / 4

df["Behavior_metabolism_score"] = (
    df["Methanobrevibacter"]
    + df["Intestinimonas"]
    + (1 - df["Nicotinate_metabolism"])
    + (1 - df["Caffeine_metabolism"])
) / 4

df["Drug_microbiome_interaction"] = (
    df["Entacapone"] * df["Microbiome_dysbiosis_score"]
)

# =========================
# 9. Data dictionary
# =========================

data_dictionary = pd.DataFrame({
    "Column": [
        "Entacapone", "ICD", "GI_side_effect",
        "Methanobrevibacter", "Intestinimonas",
        "Faecalibacterium", "Lactobacillus", "Bifidobacterium",
        "Nicotinate_metabolism", "Caffeine_metabolism",
        "Xenobiotic_degradation", "Indole_metabolism", "Amino_acid_metabolism",
        "Microbiome_dysbiosis_score", "Behavior_metabolism_score",
        "Drug_microbiome_interaction"
    ],
    "Meaning": [
        "Whether patient received entacapone",
        "Proxy label for impulse control disorder",
        "Proxy label for GI side effects",
        "ICD-associated genus; enriched in ICD group",
        "ICD-associated genus/species-level proxy",
        "Decreased under entacapone in related paper",
        "Decreased under entacapone in related paper",
        "Increased under entacapone in related paper",
        "Reduced in ICD-associated functional analysis",
        "Reduced in ICD-associated functional analysis",
        "Drug/xenobiotic metabolism-related pathway",
        "Tryptophan/serotonin-related gut-brain pathway proxy",
        "Increased in entacapone-associated functional analysis",
        "Composite proxy of microbiome imbalance",
        "Composite proxy related to impulsivity pathway",
        "Interaction term between drug exposure and dysbiosis"
    ],
    "Type": [
        "Observed or generated from group ratio",
        "Generated proxy", "Generated proxy",
        "Generated proxy", "Generated proxy",
        "Generated proxy", "Generated proxy", "Generated proxy",
        "Generated proxy", "Generated proxy",
        "Generated proxy", "Generated proxy", "Generated proxy",
        "Engineered feature", "Engineered feature", "Engineered feature"
    ]
})

# =========================
# 10. Save output
# =========================

output_file = "expanded_pd_microbiome_dataset.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Enhanced_Dataset", index=False)
    data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)

print(f"Saved to {output_file}")
print(df.shape)
print(df[["Entacapone", "ICD", "GI_side_effect"]].mean())