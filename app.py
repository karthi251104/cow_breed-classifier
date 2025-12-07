from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (224, 224)

# ---------------- CLASS NAMES ----------------
CLASS_NAMES = [...]
# (keep your full list unchanged)

# ---------------- LOAD TFLITE MODEL (with better error handling) ----------------
print("Loading TFLite model from:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}! Did you forget to add it to Git?")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✔ TFLite model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- PREDICTION FUNCTION ----------------
def predict_breed(image):
    arr = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# HEALTH CHECK ENDPOINT (this is the key fix!)
@app.route("/health")
def health():
    return "OK", 200  # Plain text, 200 status → Render loves this

# Optional: keep your nice JSON root if you want
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed TFLite API Running!", "health": "OK"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Invalid image"}), 400
    
    breed, confidence = predict_breed(img)
    return jsonify({
        "breed": breed,
        "confidence": round(confidence, 4)
    })

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Debug off in production for speed + security
    app.run(host="0.0.0.0", port=port, debug=False)