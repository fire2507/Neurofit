import pandas as pd
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =====================
# LOAD DATA
# =====================
DATA_PATH = r"C:\Users\user\EEG\data"

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

df = pd.concat(
    [pd.read_csv(os.path.join(DATA_PATH, f)) for f in files],
    ignore_index=True
)

# =====================
# CREATE BAND AVERAGES
# =====================
bands = ['alpha','beta','gamma','delta','theta']

for band in bands:
    df[band] = df[[f"{band}{i}" for i in range(4)]].mean(axis=1)

# =====================
# CREATE LABELS (LOGIC BASED)
# =====================
def label_state(row):
    if row['beta'] > row['alpha'] and row['beta'] > row['theta']:
        return "Focused"
    elif row['theta'] > row['beta']:
        return "Distracted"
    elif row['alpha'] > row['beta']:
        return "Creative"
    else:
        return "Neutral"

df['state'] = df.apply(label_state, axis=1)

# =====================
# FEATURES
# =====================
X = df[['delta','theta','alpha','beta','gamma']]
y = df['state']

# =====================
# TRAIN MODEL
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# =====================
# DASHBOARD
# =====================
st.title("🧠 Real-Time Brain State Detector")

idx = st.slider("Select Sample", 0, len(X)-1, 0)

sample = X.iloc[idx:idx+1]
prediction = model.predict(sample)[0]

st.subheader(f"Predicted State: {prediction}")

# Plot bands
fig, ax = plt.subplots()
ax.bar(sample.columns, sample.values[0])
st.pyplot(fig)