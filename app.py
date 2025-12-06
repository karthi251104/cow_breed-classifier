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

MODEL_PATH = "model.tflite"
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
print("Checking model file...")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Full path:", os.path.abspath(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")


# =============================
# FLASK APP
# =============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# =============================
# PREPROCESS FUNCTION
# =============================
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


# =============================
# PREDICT API
# =============================
@app.route("/predict", methods=["POST"])
def predict_api():

    print("FILES:", request.files)

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file.stream)
    except:
        return jsonify({"error": "Invalid image"}), 400

    arr = preprocess(img)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    print("Model output shape:", output.shape)
    print("Model output:", output)

    pred_index = int(np.argmax(output))
    confidence = float(output[pred_index])

    # VALIDATE CLASS COUNT
    model_classes = len(output)
    breed_classes = len(BREEDS)

    if model_classes != breed_classes:
        return jsonify({
            "error": "Model output count does NOT match breed list!",
            "model_classes": model_classes,
            "breed_list_classes": breed_classes
        }), 500

    predicted_breed = BREEDS[pred_index]

    return jsonify({
        "breed": predicted_breed,
        "confidence": round(confidence * 100, 2)
    })


# =============================
# RUN SERVER LOCALLY
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
