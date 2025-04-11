import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Vocabulaire
with open("model/location_vocab.txt") as f:
    location_vocab = [line.strip() for line in f.readlines()]

with open("model/room_type_vocab.txt") as f:
    room_type_vocab = [line.strip() for line in f.readlines()]

room_type_map = {
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "DiningRoom": "dining",
    "Entryway": "entryway",
    "Hallway": "hallway",
    "Kitchen": "kitchen",
    "LivingRoom": "living",
    "MainEntryway": "entryway",
    "Office": "office"
}

def extract_room_type(location):
    for prefix, rtype in room_type_map.items():
        if location.startswith(prefix):
            return rtype
    return "other"

# Dataset
df = pd.read_csv("data/test_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df_motion = df[df["device"].str.contains("Control4-Motion", case=False)].copy()
df_motion = df_motion.dropna(subset=["timestamp"])
df_motion = df_motion[df_motion["location"].isin(location_vocab)]

# Features
df_motion["light_on"] = df_motion["status"].apply(lambda x: 1 if isinstance(x, str) and x.upper() == "ON" else 0)
df_motion["hour_float"] = df_motion["timestamp"].dt.hour + df_motion["timestamp"].dt.minute / 60
df_motion["is_night"] = df_motion["timestamp"].dt.hour.apply(lambda h: 1 if h >= 19 or h < 6 else 0)
df_motion["previous_light_on"] = df_motion["light_on"].shift(1).fillna(0).astype(int)
df_motion["location_enc"] = df_motion["location"].apply(lambda loc: location_vocab.index(loc))
df_motion["room_type"] = df_motion["location"].apply(extract_room_type)
df_motion["room_type_enc"] = df_motion["room_type"].apply(lambda x: room_type_vocab.index(x))

# Features continues
hour_sin = np.sin(2 * np.pi * df_motion["hour_float"] / 24)
hour_cos = np.cos(2 * np.pi * df_motion["hour_float"] / 24)
X_cont = np.stack([
    hour_sin,
    hour_cos,
    df_motion["location_enc"],
    df_motion["room_type_enc"]
], axis=-1)

# Binaires
X_binary = df_motion[["is_night", "previous_light_on"]].values

# Normalisation
scaler = joblib.load("model/scaler.save")
X_scaled = scaler.transform(X_cont)

# Fusion finale
X = np.concatenate([X_scaled, X_binary], axis=1)
y = df_motion["light_on"].values

# Séquences temporelles
def create_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y)

# Prédictions
model = load_model("model/lstm_model.keras", compile=False)
y_pred_probs = model.predict(X_seq, verbose=1)
y_pred = (y_pred_probs > 0.45).astype(int)

# Évaluation
print("\n📊 Rapport de classification :")
print(classification_report(y_seq, y_pred, target_names=["Off", "On"]))
print("\n🌟 F1 Score global:", round(f1_score(y_seq, y_pred), 4))

# Matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_seq, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Off", "On"], yticklabels=["Off", "On"])
plt.xlabel("Prediction")
plt.ylabel("Réel")
plt.title("Matrice de confusion - Dataset Externe")
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/confusion_matrix_external.png")
plt.show()
