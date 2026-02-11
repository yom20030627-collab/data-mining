import pandas as pd
import numpy as np

df = pd.read_csv("simulated_hospital_large_dataset.csv")

features = ["age", "BMI", "smoking", "alcohol", "hypertension", "diabetes", "hyperlipidemia", "CKD", "CAD", "stroke_history"]

X = df[features]
y = df["mortality"]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

 # Convert probabilities to binary predictions
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class (mortality)
y_pred = (y_prob >= 0.4).astype(int)  # Threshold at 0.2
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.show()