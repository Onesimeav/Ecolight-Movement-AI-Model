import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Charger les données brutes
df = pd.read_csv('data/dataset.csv', header=None, names=['timestamp', 'location', 'status', 'device'])
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce')

# Séparer les capteurs de mouvement
df_motion = df[df["device"].str.contains("Control4-Motion", case=False)].copy()

# Ajouter les colonnes nécessaires
df_motion["light_on"] = df_motion["status"].apply(lambda x: 1 if isinstance(x, str) and x.upper() == "ON" else 0)
df_motion["hour_float"] = df_motion["timestamp"].dt.hour + df_motion["timestamp"].dt.minute / 60
df_motion["day_of_week"] = df_motion["timestamp"].dt.dayofweek
df_motion["is_night"] = df_motion["timestamp"].dt.hour.apply(lambda h: 1 if h >= 19 or h < 6 else 0)
df_motion["previous_light_on"] = df_motion["light_on"].shift(1).fillna(0).astype(int)

# Encodage des pièces
df_motion["location_enc"] = df_motion["location"].astype("category").cat.codes
location_vocab = df_motion["location"].astype("category").cat.categories

# Encodage du type de pièce
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

def extract_room_type(loc):
    for prefix, val in room_type_map.items():
        if loc.startswith(prefix):
            return val
    return "other"

df_motion["room_type"] = df_motion["location"].apply(extract_room_type)
df_motion["room_type_enc"] = df_motion["room_type"].astype("category").cat.codes
room_type_vocab = df_motion["room_type"].astype("category").cat.categories

# Final features
features = ['hour_float', 'location_enc', 'day_of_week', 'room_type_enc', 'is_night', 'previous_light_on']
df_final = df_motion[features + ['light_on']].dropna()

# Normalisation (sur les features continues uniquement)
X = df_final[features].values
y = df_final['light_on'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[:, :4])  # only on continuous: hour, loc_enc, dow, room_type
X_combined = np.concatenate([X_scaled, X[:, 4:]], axis=1)  # concat with binary features

# Création des séquences
def create_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_combined, y)

# Sauvegarde
np.savez("data/prepared_from_txt.npz", X=X_seq, y=y_seq)
joblib.dump(scaler, "model/scaler.save")

with open("model/location_vocab.txt", "w") as f:
    for loc in location_vocab:
        f.write(f"{loc}\n")

with open("model/room_type_vocab.txt", "w") as f:
    for rtype in room_type_vocab:
        f.write(f"{rtype}\n")

print(f"✅ Données préparées : {X_seq.shape[0]} séquences prêtes.")
