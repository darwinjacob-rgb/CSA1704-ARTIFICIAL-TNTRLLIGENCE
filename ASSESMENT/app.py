from pathlib import Path
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "crop_disease_cnn.keras"
CLASS_PATH = BASE / "metrics" / "classes.json"
METRICS_PATH = BASE / "metrics" / "metrics.json"
HISTORY_PATH = BASE / "metrics" / "training_history.json"
SAMPLES_PATH = BASE / "metrics" / "validation_samples.json"

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model(MODEL_PATH) if MODEL_PATH.exists() else None
classes = json.loads(CLASS_PATH.read_text()) if CLASS_PATH.exists() else []

@app.get("/health")
def health():
    return jsonify({"status": "online", "model_loaded": model is not None,
                    "metrics_available": METRICS_PATH.exists()})

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "No evaluation metrics yet. Train and evaluate the CNN first."}), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))

@app.get("/history")
def history():
    if not HISTORY_PATH.exists():
        return jsonify({"error": "Training history not available."}), 404
    return jsonify(json.loads(HISTORY_PATH.read_text()))

@app.get("/validation-samples")
def validation_samples():
    if not SAMPLES_PATH.exists():
        return jsonify({"error": "Validation samples are not available yet."}), 404
    return jsonify(json.loads(SAMPLES_PATH.read_text()))

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "CNN model not found. Run training first."}), 503
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image supplied."}), 400
    try:
        img = Image.open(f).convert("RGB").resize((224, 224))
        x = np.asarray(img, dtype=np.float32) / 255.0
        probs = model.predict(x[None, ...], verbose=0)[0]
        i = int(np.argmax(probs))
        label = classes[i] if i < len(classes) else str(i)
        parts = label.split("___", 1)
        crop = parts[0].replace("_", " ") if parts else label
        disease = parts[1].replace("_", " ") if len(parts) > 1 else label.replace("_", " ")
        return jsonify({"class_id": i, "label": label, "crop": crop,
                        "disease": disease, "confidence": float(probs[i])})
    except Exception as exc:
        return jsonify({"error": f"Unable to analyze image: {exc}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
