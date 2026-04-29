import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "vanet_traffic_data.csv"
MODEL_FILE = "model.pkl"
COLUMNS_FILE = "columns.pkl"
LABEL_MAP_FILE = "label_map.pkl"

LABEL_MAP = {
    "Low": 0,
    "Medium": 1,
    "Heavy": 2,
}

CLASS_NAMES = {
    0: "Low Traffic",
    1: "Medium Traffic",
    2: "Heavy Traffic",
}

FEATURE_COLUMNS = [
    "avg_speed_kmph",
    "density_veh_per_km",
    "avg_wait_time_s",
    "occupancy_pct",
    "flow_veh_per_hr",
    "queue_length_veh",
    "avg_accel_ms2",
    "weather_factor",
]


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[col for col in ["timestamp", "road_segment_id"] if col in df.columns], errors="ignore")

    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df["label"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df


def train_model(X, y):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model


def main():
    print("Loading dataset...")
    df = load_data(DATA_FILE)

    print("Cleaning data...")
    df = clean_data(df)

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    labels = sorted(y_test.unique())
    target_names = [CLASS_NAMES[i] for i in labels]
    print("Classification report:\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))

    print("Saving artifacts...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(FEATURE_COLUMNS, COLUMNS_FILE)
    joblib.dump(CLASS_NAMES, LABEL_MAP_FILE)

    print(f"Saved {MODEL_FILE}, {COLUMNS_FILE}, and {LABEL_MAP_FILE}.")


if __name__ == "__main__":
    main()
