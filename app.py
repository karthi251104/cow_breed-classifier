import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------
# CONFIG
# ----------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "model.tflite"   # <---- YOUR MODEL NAME
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

# ----------------------------
# LOAD TFLITE MODEL
# ----------------------------
print("Checking model file...")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Full path:", os.path.abspath(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")

# ----------------------------
# FLASK APP
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------
# PREDICT ENDPOINT
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():

    print("FILES:", request.files)
    print("FORM:", request.f)
