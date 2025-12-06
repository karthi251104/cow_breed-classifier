# app.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "model.tflite"           # Make sure this file exists in repo!
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

# ==========================
# LOAD MODEL ONCE
# ==========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)
CORS(app)

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Remove the next line if your model was trained with /255.0
    # arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    # Run model
    input_data = preprocess(img)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    # Top 5
    top5_idx = np.argsort(output)[-5:][::-1]
    top5 = [
        {
            "label": BREEDS[i],
            "confidence": round(float(output[i]), 4)   # 0.9821 format
        }
        for i in top5_idx
    ]

    # Main result
    main_breed = top5[0]["label"]
    main_conf = round(top5[0]["confidence"] * 100, 2)

    return jsonify({
        "breed": main_breed,
        "confidence": main_conf,
        "top_predictions": top5
    })

@app.route("/")
def home():
    return "Cow Breed Classifier API – Ready!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)