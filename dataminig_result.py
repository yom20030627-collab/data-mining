import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 讀資料
df = pd.read_excel("C:/Users/user/Desktop/expanded_pd_microbiome_dataset.xlsx")
df["ICD"] = (df["ICD"] > 0.15).astype(int)
print(df["ICD"].value_counts())

# 選 feature（不要用 label）
features = [
    "Methanobrevibacter", "Intestinimonas",
    "Faecalibacterium", "Lactobacillus",
    "Bifidobacterium",
    "Nicotinate_metabolism", "Caffeine_metabolism",
    "Amino_acid_metabolism",
    "Entacapone"
]

X = df[features]
y = df["ICD"]

# 切 train/test：保持 ICD 比例一致
# 修 label

# feature / label
X = df[features]
y = df["ICD"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 建模型：加 random_state，並處理類別不平衡
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

importances = pd.Series(model.feature_importances_, index=features).sort_values()

plt.figure(figsize=(8,5))
plt.barh(importances.index, importances.values)
plt.title("Feature Importance for ICD Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.title("PCA of Microbiome Data by ICD Status")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("K-means Clustering Result")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.title("Clustering result")
plt.show()

print("ICD distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))