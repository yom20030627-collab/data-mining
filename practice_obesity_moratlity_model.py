import pandas as pd
import numpy as np

df = pd.read_csv("simulated_hospital_vte_upgraded_dataset.csv")
df["obesity"] = (df["BMI"] >= 27).astype(int)
df["HTN-DM"] = df["hypertension"] * df["diabetes"]
df["obesity-HTN"] = df["obesity"] * df["hypertension"]
df["obesity-DM"] = df["obesity"] * df["diabetes"]
df["obesity-HTN-DM"] = df["obesity"] * df["hypertension"] * df["diabetes"]
df["obesity-HTN-DM-CKD"] = df["obesity"] * df["hypertension"] * df["diabetes"] * df["CKD"]
df["obesity-HTN-DM-CKD-CAD"] = df["obesity"] * df["hypertension"] * df["diabetes"] * df["CKD"] * df["CAD"]

features = ["age", "BMI", "smoking", "alcohol", "hypertension", "diabetes", "hyperlipidemia", "CKD", "obesity","CAD", "stroke_history", "HTN-DM", "obesity-HTN", "obesity-DM", "obesity-HTN-DM", "obesity-HTN-DM-CKD", "obesity-HTN-DM-CKD-CAD"]

X = df[features]
y = df["mortality"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class (mortality)
best_f1 = 0
best_threshold = 0
for t in np.arange(0.1, 0.9, 0.05):
    temp_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, temp_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_prob >= best_threshold).astype(int)  # Threshold at best_threshold



 # Convert probabilities to binary predictions
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("Best Threshold:", best_threshold)
print("Best F1 Score:", best_f1)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.show()