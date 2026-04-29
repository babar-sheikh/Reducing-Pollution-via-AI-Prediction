# Reducing Pollution via AI Prediction

This project predicts traffic congestion level from vehicle and road metrics. It provides a Flask web interface for real-time predictions and a training script to create the model artifacts.

## Setup

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Start the web app:
   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`

## Files

- `app.py` - Flask web application.
- `train_model.py` - Train the model from `vanet_traffic_data.csv`.
- `vanet_traffic_data.csv` - Dataset used for training.
- `templates/index.html` - Web page template.
- `static/style.css` - Basic styles for the app.
- `model.pkl`, `columns.pkl`, `label_map.pkl` - Generated model artifacts.
