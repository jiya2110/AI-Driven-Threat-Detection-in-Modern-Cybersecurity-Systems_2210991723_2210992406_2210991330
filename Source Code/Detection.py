"""
=============================================================================
 AI-DRIVEN SOCIAL MEDIA BOT DETECTION
 Hybrid Behavioral + Content + Anomaly + Graph Intelligence

 RESEARCH-GRADE IMPLEMENTATION (NIFTY-STYLE QUALITY)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os, time

# =============================================================================
# THEME (NIFTY STYLE)
# =============================================================================
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
GRID = "#30363d"
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#f78166"
PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
})

OUTPUT_DIR = "research_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    path = f"{OUTPUT_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved -> {path}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "="*70)
print(" LOADING DATASET")
print("="*70)

df = pd.read_csv("bot_detection_data.csv")
print(f"Dataset shape: {df.shape}")

# =============================================================================
# CLEANING
# =============================================================================
df = df.fillna({
    "Tweet": "",
    "Hashtags": "",
    "Mention Count": 0,
    "Retweet Count": 0,
    "Follower Count": 0
})

# =============================================================================
# FEATURE ENGINEERING (STRONG VERSION)
# =============================================================================
print("\nEngineering features...")

df["tweet_length"] = df["Tweet"].astype(str).apply(len)
df["hashtag_count"] = df["Hashtags"].apply(lambda x: len(str(x).replace(",", " ").split()))

df["engagement_ratio"] = df["Retweet Count"] / (df["Follower Count"] + 1)
df["mention_density"] = df["Mention Count"] / (df["tweet_length"] + 1)

df["followers_log"] = np.log1p(df["Follower Count"])
df["retweet_log"] = np.log1p(df["Retweet Count"])
df["mention_log"] = np.log1p(df["Mention Count"])

df["has_url"] = df["Tweet"].str.contains("http").astype(int)
df["has_caps"] = df["Tweet"].str.contains(r"[A-Z]{3,}").astype(int)
df["exclamations"] = df["Tweet"].str.count("!")

df["verified"] = df["Verified"].astype(int)
df["Bot Label"] = df["Bot Label"].astype(int)

features = [
    "followers_log","retweet_log","mention_log",
    "engagement_ratio","mention_density",
    "tweet_length","hashtag_count",
    "has_url","has_caps","exclamations","verified"
]

X = df[features]
y = df["Bot Label"]

# =============================================================================
# SCALING
# =============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# AUTOENCODER
# =============================================================================
print("\nTraining Autoencoder...")

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16))
        self.dec = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self,x): return self.dec(self.enc(x))

model = AE(X_scaled.shape[1])
opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X_tensor = torch.FloatTensor(X_scaled)

for epoch in range(40):
    opt.zero_grad()
    out = model(X_tensor)
    loss = loss_fn(out, X_tensor)
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"  Epoch {epoch} Loss: {loss.item():.5f}")

with torch.no_grad():
    recon = model(X_tensor)
    df["ae_score"] = torch.mean((X_tensor-recon)**2, dim=1).numpy()

# =============================================================================
# ISOLATION FOREST
# =============================================================================
iso = IsolationForest(contamination=0.1)
iso.fit(X_scaled)
df["iso_score"] = -iso.decision_function(X_scaled)

# =============================================================================
# GRAPH (SIMILARITY BASED)
# =============================================================================
print("\nBuilding similarity graph...")
from sklearn.metrics.pairwise import cosine_similarity

sample = X_scaled[:1000]
sim = cosine_similarity(sample)

G = nx.Graph()
for i in range(len(sim)):
    for j in range(i+1,len(sim)):
        if sim[i][j] > 0.9:
            G.add_edge(i,j)

degrees = dict(G.degree())
df["degree"] = df.index.map(lambda x: degrees.get(x,0))

# =============================================================================
# MODEL FUNCTION
# =============================================================================
def run(X, name):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    clf = GradientBoostingClassifier()
    clf.fit(Xtr,ytr)

    yp = clf.predict(Xte)
    yp_prob = clf.predict_proba(Xte)[:,1]

    return {
        "name": name,
        "precision": precision_score(yte,yp),
        "recall": recall_score(yte,yp),
        "f1": f1_score(yte,yp),
        "roc": roc_auc_score(yte,yp_prob)
    }

# =============================================================================
# EXPERIMENTS
# =============================================================================
print("\nRunning experiments...")

results = []
results.append(run(X_scaled,"Baseline"))
results.append(run(np.hstack((X_scaled,df["ae_score"].values.reshape(-1,1))),"Hybrid AE"))
results.append(run(np.hstack((X_scaled,df["iso_score"].values.reshape(-1,1))),"Hybrid ISO"))
results.append(run(np.hstack((X_scaled,df["degree"].values.reshape(-1,1))),"Graph"))

# =============================================================================
# PRINT TABLE
# =============================================================================
print("\n" + "="*70)
print(" FINAL RESULTS ")
print("="*70)

for r in results:
    print(f"{r['name']:<12} | P:{r['precision']:.3f} R:{r['recall']:.3f} F1:{r['f1']:.3f} ROC:{r['roc']:.3f}")

# =============================================================================
# FIGURE 1 — FEATURE DISTRIBUTION
# =============================================================================
print("\n[Figure 1] Feature Distribution...")

fig, axs = plt.subplots(2,3, figsize=(14,8))
axs = axs.flatten()

for i,f in enumerate(features[:6]):
    axs[i].hist(df[f], bins=50, color=BLUE, alpha=0.7)
    axs[i].set_title(f)

plt.tight_layout()
savefig("fig1_features")

# =============================================================================
# FIGURE 2 — MODEL COMPARISON
# =============================================================================
print("[Figure 2] Model Comparison...")

labels = [r["name"] for r in results]
f1s = [r["f1"] for r in results]

plt.figure(figsize=(8,5))
plt.bar(labels,f1s,color=[BLUE,ORANGE,GREEN,PURPLE])
plt.title("F1 Score Comparison")
savefig("fig2_models")

# =============================================================================
# FIGURE 3 — ANOMALY DISTRIBUTION
# =============================================================================
plt.figure(figsize=(8,5))
sns.histplot(df["ae_score"], bins=60, kde=True)
plt.title("Autoencoder Anomaly Score")
savefig("fig3_anomaly")

# =============================================================================
# FIGURE 4 — GRAPH DEGREE
# =============================================================================
plt.figure(figsize=(8,5))
sns.histplot(df["degree"], bins=50)
plt.title("Graph Degree Distribution")
savefig("fig4_graph")

print("\nAll figures saved.")
print("="*70)
print(" DONE 🚀")
print("="*70)