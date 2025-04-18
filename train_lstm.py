import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Add, Attention
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Fonction de perte focalisée
@register_keras_serializable()
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = alpha * tf.pow(1. - p_t, gamma) * bce
        return tf.reduce_mean(focal)
    return loss

# Chargement des données
data = np.load("data/prepared_from_txt.npz")
X = data["X"]
y = data["y"]

# Vérification des dimensions
print(f"📦 X shape: {X.shape} | y shape: {y.shape}")

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Pondération des classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Construction du modèle Res-LSTM avec Attention
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1 = LSTM(64, return_sequences=True)(inputs)
lstm2 = LSTM(64, return_sequences=True)(lstm1)
res = Add()([lstm1, lstm2])
attention_out = Attention()([res, res])
out = Dense(1, activation='sigmoid')(attention_out[:, -1, :])
model = Model(inputs, out)

model.compile(
    optimizer='adam',
    loss=binary_focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    class_weight=class_weight_dict
)

# Sauvegarde du modèle
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.keras")

# Évaluation
print("\n🔍 Évaluation du modèle sur les données de validation :")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"✔️  Accuracy : {accuracy:.4f}")
print(f"✔️  Loss     : {loss:.4f}")

# Prédictions
y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.45).astype(int)

# Rapport de classification
report = classification_report(y_val, y_pred, target_names=["Off", "On"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("model/classification_report.csv")
print("\n📊 Rapport de classification :")
print(report_df)

# F1 global
f1 = f1_score(y_val, y_pred)
print(f"\n🌟 F1-score global : {f1:.4f}")

# Matrice de confusion
conf_mat = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Off", "On"], yticklabels=["Off", "On"])
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

print("✅ Entraînement terminé sans la feature light_level.")