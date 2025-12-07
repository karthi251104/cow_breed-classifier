from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"
IMAGE_SIZE = (224, 224)

# ---------------- CLASS NAMES ----------------
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

# ---------------- LOAD MODEL ----------------
print("Loading TFLite model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.tflite missing in project root!")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✔ TFLite model loaded")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- PREPROCESS ----------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- PREDICT ----------------
def predict_breed(img):
    arr = preprocess_image(img)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    return CLASS_NAMES[idx], conf

# ---------------- FLASK APP ----------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed TFLite API Running!", "health": "OK"})

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# FIX: allow GET to show instructions (avoid 405)
@app.route("/predict", methods=["GET"])
def predict_info():
    return jsonify({
        "message": "Use POST with form-data: image=<file>",
        "example": "/predict (POST)"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image file"}), 400

    breed, conf = predict_breed(img)
    return jsonify({
        "breed": breed,
        "confidence": round(conf, 4)
    })

# ---------------- RUN (Render PORT) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
