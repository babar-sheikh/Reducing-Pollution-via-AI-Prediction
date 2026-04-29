from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_FILE = "model.pkl"
COLUMNS_FILE = "columns.pkl"
LABEL_MAP_FILE = "label_map.pkl"

INPUT_FIELDS = [
    ("avg_speed_kmph", "Average speed (km/h)"),
    ("density_veh_per_km", "Vehicle density (vehicles/km)"),
    ("avg_wait_time_s", "Average wait time (s)"),
    ("occupancy_pct", "Occupancy (%)"),
    ("flow_veh_per_hr", "Flow (vehicles/hr)"),
    ("queue_length_veh", "Queue length (vehicles)"),
    ("avg_accel_ms2", "Average acceleration (m/s²)"),
    ("weather_factor", "Weather factor"),
]


def load_artifacts():
    model = None
    columns = None
    label_map = None

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    if os.path.exists(COLUMNS_FILE):
        columns = joblib.load(COLUMNS_FILE)
    if os.path.exists(LABEL_MAP_FILE):
        label_map = joblib.load(LABEL_MAP_FILE)

    if label_map is None:
        label_map = {0: "Low Traffic", 1: "Medium Traffic", 2: "Heavy Traffic"}

    return model, columns, label_map


model, columns, label_map = load_artifacts()


def parse_form_data(form):
    data = {}
    errors = []

    for field_name, label in INPUT_FIELDS:
        value = form.get(field_name, "").strip()
        if value == "":
            errors.append(f"{label} is required.")
            continue

        try:
            data[field_name] = float(value)
        except ValueError:
            errors.append(f"{label} must be a valid number.")

    return data, errors


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form
    values, errors = parse_form_data(payload)

    if errors:
        return jsonify({"error": " ".join(errors)}), 400
    if model is None or columns is None:
        return jsonify({"error": "Model files are missing. Run `python train_model.py` to generate `model.pkl` and `columns.pkl`."}), 500

    df = pd.DataFrame([values])
    df = df.reindex(columns=columns, fill_value=0)
    pred = model.predict(df)[0]
    prediction = label_map.get(int(pred), "Unknown Traffic")

    return jsonify({"prediction": prediction, "values": values})


if __name__ == "__main__":
    app.run(debug=True)
