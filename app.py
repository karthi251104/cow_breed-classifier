from flask import Flask, request, jsonify
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

print("✔ Loaded class names:", CLASS_NAMES)

# ---------------- LOAD TFLITE MODEL ----------------
print("Loading TFLite model...")
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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed TFLite API Running!"})

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

# ---------------- RUN APP (RENDER COMPATIBLE PORT) ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render injects this variable
    app.run(host="0.0.0.0", port=port)
