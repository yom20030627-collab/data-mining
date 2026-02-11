import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. 讀取 CSV（下載資料夾）
df = pd.read_csv(r"C:\Users\user\Downloads\simulated_physionet_style_data.csv")

# 2. 定義臨床狀態
df["Tachycardia"] = (df["heart_rate"] > 100).astype(int)
df["Hypoxemia"] = (df["spo2"] < 92).astype(int)
df["Hypertension"] = (df["sbp"] > 140).astype(int)
df["Fever"] = (df["temperature"] > 37.5).astype(int)

states = ["Tachycardia", "Hypoxemia", "Hypertension", "Fever"]

# 3. 建立共現邊
edges = []
for pid, g in df.groupby("patient_id"):
    for _, row in g.iterrows():
        active = [s for s in states if row[s] == 1]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                edges.append((active[i], active[j]))

# 4. 統計權重
edge_df = pd.DataFrame(edges, columns=["source", "target"])
edge_weights = edge_df.value_counts().reset_index(name="weight")

# 5. 建立網絡
G = nx.Graph()
for _, r in edge_weights.iterrows():
    G.add_edge(r["source"], r["target"], weight=r["weight"])

print("number of edges:", G.number_of_edges())

# 6. 畫圖
plt.figure(figsize=(7, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, font_size=11)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)}
)

plt.title("Clinical State Co-occurrence Network")
plt.savefig("clinical_state_network.png", dpi=300, bbox_inches="tight")
plt.show()
