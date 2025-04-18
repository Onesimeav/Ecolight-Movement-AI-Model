import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the model without compilation to bypass the loss function issue
print("Loading model without compilation...")
model = load_model("model/lstm_model.keras", compile=False)
print("Model loaded successfully!")

# Print the input shape the model expects
input_shape = model.input_shape
print(f"Model expects input shape: {input_shape}")

# Load the scaler
scaler = joblib.load('model/scaler.save')
print(f"Scaler features: {scaler.n_features_in_}")

# Charger le vocabulaire des pièces
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


def convert_to_prediction_features(timestamp, location, light_level, previous_light_on=0):
    timestamp = pd.to_datetime(timestamp)
    hour_float = timestamp.hour + timestamp.minute / 60
    day_of_week = timestamp.dayofweek

    # Calculate the sine and cosine components
    hour_sin = np.sin(2 * np.pi * hour_float / 24)
    hour_cos = np.cos(2 * np.pi * hour_float / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Encode location and room type
    location_enc = location_vocab.index(location)
    room_type = extract_room_type(location)
    room_type_enc = room_type_vocab.index(room_type)

    light_level = float(light_level)
    is_night = 1 if timestamp.hour >= 19 or timestamp.hour < 6 else 0

    # Normalize the continuous features - making sure we have 7 features for the scaler
    to_scale = [[hour_sin, hour_cos, location_enc, light_level, dow_sin, dow_cos, room_type_enc]]
    scaled = scaler.transform(to_scale)[0]

    # Add binary features - making sure we have exactly 10 total features
    # The model expects 10 features, and we need to ensure we provide exactly that
    features = np.concatenate([scaled, [is_night, previous_light_on, 0]])  # Added a dummy feature to reach 10

    # Ensure we have exactly 10 features
    if len(features) != 10:
        if len(features) < 10:
            features = np.pad(features, (0, 10 - len(features)), 'constant')
        else:
            features = features[:10]

    return features

def predict_light_status(features):
    # The model expects shape (None, 20, 10), but we have a single feature vector
    # Create a properly sized input array with our features repeated
    sequence_length = 20  # from the model's expected input shape
    num_features = 10  # from the model's expected input shape

    # Check if our features match the expected number
    if len(features) != num_features:
        print(f"Warning: Feature count mismatch. Got {len(features)}, expected {num_features}")
        # Pad or truncate features to match expected size
        if len(features) < num_features:
            features = np.pad(features, (0, num_features - len(features)), 'constant')
        else:
            features = features[:num_features]

    # Create a sequence by repeating our single feature vector
    X_sequence = np.tile(features, (sequence_length, 1))

    # Reshape to match model's expected input shape: (batch_size, sequence_length, features)
    X_new = np.expand_dims(X_sequence, axis=0)  # shape = (1, 20, 10)

    # Print shapes for verification
    print(f"Input shape: {X_new.shape}")

    # Make prediction
    prediction = model.predict(X_new, verbose=0)
    return prediction[0][0] > 0.45


# Example usage:
timestamp = '2016-11-01 18:03:30.323230'
location = 'KitchenARefrigerator'
light_level = 58

# Get features
features = convert_to_prediction_features(timestamp, location, light_level, previous_light_on=0)
print(f"Features shape: {features.shape}, Features: {features}")

# Use your predict_light_status function which handles the sequence creation
prediction_result = predict_light_status(features)
print(f"Turn on the light? {prediction_result}")