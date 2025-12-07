from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (224, 224)

# ---------------- CLASS NAMES (EMBEDDED AS TEXT) ----------------
CLASS_NAMES = """
Alambadi
Amritmahal
Ayrshire
Banni
Bargur
Bhadawari
Brown_Swiss
Dangi
Deoni
Gir
Guernsey
Hallikar
Hariana
Holstein_Friesian
Jaffrabadi
Jersey
Kangayam
Kankrej
Kasargod
Kenkatha
Kherigarh
Khillari
Krishna_Valley
Malnad_gidda
Mehsana
Murrah
Nagori
Nagpuri
Nili_Ravi
Nimari
Ongole
Pulikulam
Rathi
Red_Dane
Red_Sindhi
Sahiwal
Surti
Tharparkar
Toda
Umblachery
Vechur
""".strip().split("\n")

print("✔ Loaded class names:", CLASS_NAMES)

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✔ Model loaded successfully!")

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- PREDICT FUNCTION ----------------
def predict_breed(image):
    arr = preprocess_image(image)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return CLASS_NAMES[idx], confidence

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    breed, confidence = predict_breed(img)

    return jsonify({
        "breed": breed,
        "confidence": confidence
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed Classifier API Running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
