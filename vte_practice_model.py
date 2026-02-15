from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# 1️⃣ 讀資料
df = pd.read_csv("simulated_hospital_vte_upgraded_dataset.csv")

# 2️⃣ 建立特徵
df["obesity"] = (df["BMI"] >= 27).astype(int)

features = [
    "age","cancer","D_dimer","operation_duration_min",
    "blood_transfusion","nonambulatory_preop","spine_fusion",
    "pharm_prophylaxis","mech_prophylaxis",
    "BMI","smoking","diabetes","CKD",
    "hyperlipidemia","hypertension","obesity","CAD","stroke_history"
]

X = df[features].fillna(df[features].median())
y = df["VTE"]

print("VTE incidence:", y.mean())

# 3️⃣ 切分資料（這一步一定要在 model.fit 之前）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ 建立模型
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

# 5️⃣ 訓練
rf_model.fit(X_train, y_train)

# 6️⃣ 預測
y_prob = rf_model.predict_proba(X_test)[:,1]

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, classification_report, roc_auc_score

RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced'
)
model = rf_model
model.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class (mortality)
y_pred = (y_prob >= 0.5).astype(int)  # Threshold at 0.5
y_prob_log = log_model.predict_proba(X_test)[:,1]
fpr2, tpr2, _ = roc_curve(y_test, y_prob_log)

plt.plot(fpr2, tpr2)



from sklearn.metrics import roc_auc_score, average_precision_score

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("PR AUC:", average_precision_score(y_test, y_prob))


 # Convert probabilities to binary predictions
print(classification_report(y_test,y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print(df["VTE"].value_counts(normalize=True))
print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_prob)
roc_curve = roc_curve(y_test, y_prob)
plt.plot(roc_curve[0], roc_curve[1])

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.show()