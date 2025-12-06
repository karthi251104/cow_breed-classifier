import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

MODEL_PATH = "Best_Cattle_Breed.h5"
IMAGE_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

CLASS_NAMES = [
    "Umblachery", "Tharparkar", "Toda", "Sahiwal", "Surti", "Red_Dane",
    "Rathi", "Pulikulam", "Ongole", "Nimari", "Nagpuri", "Nili_Ravi",
    "Nagori", "Murrah", "Mehsana", "Malnad_gidda", "Krishna_Valley",
    "Khillari", "Kasargod", "Kenkatha", "Kherigarh", "Kankrej",
    "Kangayam", "Jaffrabadi", "Jersey", "Holstein_Friesian", "Hariana",
    "Hallikar", "Guernsey", "Gir", "Deoni", "Dangi", "Bhadawari",
    "Brown_Swiss", "Bargur", "Banni", "Ayrshire", "Amritmahal",
    "Alambadi"
]

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.route("/")
def home():
    return jsonify({"status": "API running"}), 200

# 🟢 SUPPORTS BOTH: image AND generic file upload
@app.route("/predict", methods=["POST"])
def predict_api():
    # Check for both keys: image OR file
    file = None

    if "image" in request.files:
        file = request.files["image"]
    elif "file" in request.files:
        file = request.files["file"]
    else:
        return jsonify({"error": "No image/file provided"}), 400

    # Check empty file
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    # Try opening file as image
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Uploaded file is not an image"}), 400

    arr = preprocess_image(img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return jsonify({
        "breed": CLASS_NAMES[idx],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
