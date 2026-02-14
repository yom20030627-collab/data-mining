import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx  

df = pd.read_csv("simulated_hospital_large_dataset.csv")
df["obesity"] = (df["BMI"] >= 27).astype(int)
disease = ["hypertension","diabetes", "obesity","hyperlipidemia", "CKD","CAD","stroke_history"]
print(df[disease].head())
print(df[disease].mean().sort_values(ascending=False))  

X = df[disease].astype(int).values  #(N, num_diseases)
co_mat = X.T @ X  #(num_diseases, num_diseases)

co_df = pd.DataFrame(co_mat, index=disease, columns=disease)
print(co_df)

from itertools import combinations
def phi_coefficient(a, b):
    # a, b: binary vectors (0/1)
    n11 = ((a==1) & (b==1)).sum()
    n10 = ((a==1) & (b==0)).sum()
    n01 = ((a==0) & (b==1)).sum()
    n00 = ((a==0) & (b==0)).sum()
    denom = np.sqrt((n11+n10)*(n11+n01)*(n10+n00)*(n01+n00))
    return 0 if denom == 0 else (n11*n00 - n10*n01) / denom

edges = [] 
for d1, d2 in combinations(disease, 2):
    w = phi_coefficient(df[d1].values, df[d2].values)
    edges.append((d1, d2, w))

    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)
    print(edges_sorted[:10])


G = nx.Graph()
prev = df[disease].mean()
for d in disease:
    G.add_node(d, prevalence=prev[d])  

    THRESH = 0.005
    for d1, d2, w in edges:
        if abs(w) >= THRESH:
            G.add_edge(d1, d2, weight=abs(w), sign=np.sign(w))

pos = nx.spring_layout(G, weight= 'weight',seed=103)
edge_colors = []
edge_widths = []
node_sizes = [G.nodes[d]['prevalence']*5000 for d in G.nodes]
edge_labels = {(u,v): f"{G[u][v]['weight']:.2f}" for u,v in G.edges}

for u,v in G.edges:
    w = G[u][v]['weight']
    edge_widths.append(w*10)

    if w > 0.05:
        edge_colors.append('blue')
    elif w > 0.02:
            edge_colors.append('red')
    else:
            edge_colors.append('lightcoral')

plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8)
nx .draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=8)

deg = nx.degree_centrality(G)
print("Degree Centrality:")
print(sorted(deg.items(), key=lambda x: x[1], reverse=True))

bet = nx.betweenness_centrality(G)
print("\nBetweenness Centrality:")
print(sorted(bet.items(), key=lambda x: x[1], reverse=True))

plt.title("Disease Comorbidity Network")
plt.axis('off')
plt.tight_layout()
plt.show()