from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import datetime
import io
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__, template_folder='templates')

# Load model and scaler
model = load_model("model/lstm_model.keras", compile=False)
scaler = joblib.load("model/scaler.save")

# Load vocabularies
with open("model/location_vocab.txt") as f:
    location_vocab = [line.strip() for line in f.readlines()]

with open("model/room_type_vocab.txt") as f:
    room_type_vocab = [line.strip() for line in f.readlines()]

# Room type mapping
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

room_types_used = list(set(room_type_map.values()))

def extract_room_type(location):
    for prefix, rtype in room_type_map.items():
        if location.startswith(prefix):
            return rtype
    return "other"

def generate_features(timestamp, location, previous_light_on=0):
    try:
        hour_float = timestamp.hour + timestamp.minute / 60
        day_of_week = timestamp.weekday()
        hour_sin = np.sin(2 * np.pi * hour_float / 24)
        hour_cos = np.cos(2 * np.pi * hour_float / 24)

        location_enc = location_vocab.index(location)
        room_type = extract_room_type(location)
        room_type_enc = room_type_vocab.index(room_type)

        # On ne garde que 4 features pour le scaler
        to_scale = [[hour_sin, hour_cos, location_enc, room_type_enc]]
        scaled = scaler.transform(to_scale)[0]  # ← doit renvoyer 4 valeurs

        is_night = 1 if timestamp.hour >= 19 or timestamp.hour < 6 else 0

        # Final features = 4 scalées + 2 binaires = 6 au total
        features = np.concatenate([scaled, [is_night, previous_light_on]])
        return features
    except Exception as e:
        print(f"[ERROR] Feature generation failed: {e}")
        return np.zeros(6)  # fallback to correct shape

def predict_sequence(room_type, start_time):
    results = []
    matching_locations = [loc for loc in location_vocab if extract_room_type(loc) == room_type]

    for location in matching_locations:
        for minutes_ahead in range(0, 121, 6):
            future_time = start_time + datetime.timedelta(minutes=minutes_ahead)
            features = generate_features(future_time, location)
            X_seq = np.tile(features, (20, 1))
            X_input = np.expand_dims(X_seq, axis=0)
            try:
                proba = model.predict(X_input, verbose=0)[0][0]
            except Exception as e:
                print(f"[ERROR] Prediction failed at {future_time}: {e}")
                proba = 0.0

            results.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "location": location,
                "predicted_on": bool(proba > 0.45),
                "confidence": float(round(proba, 4))
            })
    return results

@app.route("/")
def index():
    return render_template("index.html", room_types=sorted(set(room_type_map.values())))

@app.route("/predict_plan", methods=["POST"])
def predict_plan():
    room_type = request.form.get("room_type")
    start_time_str = request.form.get("start_time")
    try:
        start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M")
    except Exception:
        start_time = datetime.datetime.now()

    results = predict_sequence(room_type, start_time)

    # Générer un CSV temporaire
    csv_file = io.StringIO()
    writer = csv.DictWriter(csv_file, fieldnames=["time", "location", "predicted_on", "confidence"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    csv_file.seek(0)
    with open("static/plan.csv", "w", encoding="utf-8") as f:
        f.write(csv_file.getvalue())

    return render_template("results.html", results=results, room_type=room_type)

if __name__ == '__main__':
    app.run(debug=True, port=5000)