# app.py (or main.py)
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# =============================
# CONFIG
# =============================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MODEL_PATH = "model.tflite"  # ← Make sure this matches your actual filename!
IMAGE_SIZE = (224, 224)

BREEDS = [
    "Umblachery","Tharparkar","Toda","Sahiwal","Surti","Red_Dane",
    "Rathi","Pulikulam","Ongole","Nimari","Nagpuri","Nili_Ravi",
    "Nagori","Murrah","Mehsana","Malnad_gidda","Krishna_Valley",
    "Khillari","Kasargod","Kenkatha","Kherigarh","Kankrej",
    "Kangayam","Jaffrabadi","Jersey","Holstein_Friesian",
    "Hariana","Hallikar","Guernsey","Gir","Deoni","Dangi",
    "Bhadawari","Brown_Swiss","Bargur","Banni","Ayrshire",
    "Amritmahal","Alambadi"
]

# =============================
# LOAD TFLITE MODEL
# =============================
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess
    input_data = preprocess(img)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # Get top 5 predictions
    top5_idx = np.argsort(output_data)[-5:][::-1]
    top5_conf = output_data[top5_idx]
    top5_breeds = [BREEDS[i] for i in top5_idx]

    # Format exactly like your Angular frontend expects
    top_predictions = [
        {"label": breed, "confidence": round(float(conf), 4)}
        for breed, conf in zip(top5_breeds, top5_conf)
    ]

    return jsonify({
        "top_predictions": top_predictions
        # Optional: keep these for debugging
        # "predicted_breed": top_predictions[0]["label"],
        # "confidence": f"{top_predictions[0]['confidence']*100:.2f}%"
    })

# Health check
@app.route("/")
def home():
    return "Cow Breed Classifier API is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
