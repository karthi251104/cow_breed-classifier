# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- This enables CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)


CORS(app, resources={r"/predict": {"origins": "*"}, r"/": {"origins": "*"}})

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (224, 224)
TOP_K = 5  # Return top 5 predictions

# ---------------- CLASS NAMES (41 Indian + International breeds) ----------------
CLASS_NAMES = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", "Bhadawari", "Brown_Swiss",
    "Dangi", "Deoni", "Gir", "Guernsey", "Hallikar", "Hariana", "Holstein_Friesian",
    "Jaffrabadi", "Jersey", "Kangayam", "Kankrej", "Kasargod", "Kenkatha", "Kherigarh",
    "Khillari", "Krishna_Valley", "Malnad_gidda", "Mehsana", "Murrah", "Nagori",
    "Nagpuri", "Nili_Ravi", "Nimari", "Ongole", "Pulikulam", "Rathi", "Red_Dane",
    "Red_Sindhi", "Sahiwal", "Surti", "Tharparkar", "Toda", "Umblachery", "Vechur"
]

print(f"Loaded {len(CLASS_NAMES)} class names")

# ---------------- LOAD TFLite MODEL ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("TFLite model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 224, 224, 3)
    return arr

# ---------------- PREDICTION WITH TOP-5 ----------------
def get_top_predictions(image: Image.Image):
    input_array = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get top K indices
    top_indices = np.argsort(predictions)[-TOP_K:][::-1]
    top_preds = [
        {
            "label": CLASS_NAMES[idx],
            "confidence": float(predictions[idx])
        }
        for idx in top_indices
    ]
    return top_preds

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Cow Breed Classifier API is Running!",
        "breeds_count": len(CLASS_NAMES),
        "model": "EfficientNetV2 + TFLite",
        "top_k": TOP_K
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send file with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream)
        top_predictions = get_top_predictions(image)

        return jsonify({
            "success": True,
            "top_predictions": top_predictions,
            "predicted_breed": top_predictions[0]["label"],
            "confidence": round(top_predictions[0]["confidence"] * 100, 2)
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": "Invalid or corrupted image"}), 400

# ---------------- START SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)