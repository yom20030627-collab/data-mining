import pandas as pd
import numpy as np

df = pd.read_csv("simulated_hospital_large_dataset.csv")
df ["obesity"] = (df["BMI"] >= 27).astype(int)

np.random.seed(42)

risk_score = (1.5*df["hypertension"] + 1.8*df["diabetes"] + 1.3*df["obesity"] + 0.5*df["hyperlipidemia"] + 0.5*df["CKD"] + 0.5*df["CAD"] + 0.5*df["stroke_history"] + 0.7*df["smoking"] + 0.3*df["alcohol"] + 0.02*df["age"])
risk_score += 0.8 * (df["hypertension"] * df["diabetes"])  # Interaction effect
noise = np.random.normal(0, 1, size=len(df))
risk_score += 3 * noise  # Add noise to make it more realistic
prob = 1 / (1 + np.exp(-risk_score))
df["thrombosis_event"] = np.random.binomial(1, prob)
df["HTN-DM"] = df["hypertension"] * df["diabetes"]

features = ["age","HTN-DM","BMI", "smoking", "alcohol", "hypertension", "diabetes", "hyperlipidemia", "CKD", "obesity","CAD", "stroke_history"]

X = df[features]
y = df["thrombosis_event"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

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
print(df["thrombosis_event"].value_counts(normalize=True))
print("Logistic Regression ROC AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.show()