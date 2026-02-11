import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('simulated_physionet_style_data.csv')

# Calculate correlation between vital signs
vital_signs = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
correlation_matrix = df[vital_signs].corr()

# Create a network graph
G = nx.Graph()

# Add nodes for each vital sign
for vital_sign in vital_signs:
    G.add_node(vital_sign)

# Add edges based on correlation strength (threshold > 0.3)
threshold = 0.3
for i, vital_1 in enumerate(vital_signs):
    for j, vital_2 in enumerate(vital_signs):
        if i < j:
            corr_value = correlation_matrix.loc[vital_1, vital_2]
            if abs(corr_value) > threshold:
                G.add_edge(vital_1, vital_2, weight=abs(corr_value))

# Visualize the network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50)

# Draw network
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Draw edges with thickness based on correlation
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], alpha=0.6)

plt.title('Disease Network Map: Vital Signs Correlation Network', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('disease_network_map.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation matrix
print("Vital Signs Correlation Matrix:")
print(correlation_matrix)

# Identify patient risk groups
def calculate_risk_score(row):
    """Calculate composite risk score based on vital sign abnormalities"""
    risk_score = 0
    
    # Abnormal ranges (typical clinical thresholds)
    if row['heart_rate'] > 100 or row['heart_rate'] < 60:
        risk_score += 1
    if row['sbp'] > 140 or row['sbp'] < 90:
        risk_score += 1
    if row['dbp'] > 90 or row['dbp'] < 60:
        risk_score += 1
    if row['resp_rate'] > 20 or row['resp_rate'] < 12:
        risk_score += 1
    if row['spo2'] < 95:
        risk_score += 2  # Critical parameter
    if row['temperature'] > 38.5 or row['temperature'] < 36:
        risk_score += 1
    
    return risk_score

df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# Analyze risk by patient
patient_risk = df.groupby('patient_id').agg({
    'risk_score': 'mean',
    'age': 'first',
    'gender': 'first'
}).round(2)

print("\nPatient Risk Scores:")
print(patient_risk.head(20))

# Create disease-patient network
plt.figure(figsize=(14, 10))

# Identify high-risk patients
high_risk_patients = patient_risk[patient_risk['risk_score'] > 2].index.tolist()

print(f"\nHigh-risk patients (risk_score > 2): {high_risk_patients}")
