import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Ensure a sample dataset exists so training doesn't fail
if not os.path.exists("dataset.csv"):
    data_dict = {
        'followers': [500, 1000, 5000, 10000, 50000, 100000, 200, 1500, 8000, 25000],
        'hashtags': [5, 10, 8, 12, 4, 6, 2, 9, 15, 7],
        'likes': [50, 150, 800, 2000, 15000, 35000, 10, 200, 1200, 8000],
        'comments': [5, 15, 80, 250, 1000, 5000, 1, 25, 150, 900],
        'viral': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    }
    pd.DataFrame(data_dict).to_csv("dataset.csv", index=False)

data = pd.read_csv("dataset.csv")

X = data[['followers', 'hashtags', 'likes', 'comments']]
y = data['viral']

# Feature scaling to normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42) # Removes random variations
model.fit(X_scaled, y)

# Save both the model and the scaler
with open("model.pkl", "wb") as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

print("Model and Scaler trained successfully!")