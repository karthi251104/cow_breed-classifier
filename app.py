from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.tflite"

# -----------------------
# Load TFLite model
# -----------------------
print("Checking model file...")
print("Model exists:", os.path.exists(MODEL_PATH))
print("Full path:", os.path.abspath(MODEL_PATH))

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully!")
except Exception as e:
    print("❌ FAILED to load TFLite model:", e)
    raise e

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1]

# -----------------------
# FIXED BREED LABELS
# -----------------------
BREEDS = [
    "Umblachery", "Tharparkar", "Toda", "Sahiwal", "Surti",
    "Red_Dane", "Rathi", "Pulikulam", "Ongole", "Nimari",
    "Nagpuri", "Nili_Ravi", "Nagori", "Murrah", "Mehsana",
    "Malnad_gidda", "Krishna_Valley", "Khillari", "Kasargod",
    "Kenkatha", "Kherigarh", "Kankrej", "Kangayam", "Jaffrabadi",
    "Jersey", "Holstein_Friesian", "Hariana", "Hallikar",
    "Guernsey", "Gir", "Deoni", "Dangi", "Bhadawari",
    "Brown_Swiss", "Bargur", "Banni", "Ayrshire", "Amritmahal",
    "Alambadi"
]

# -----------------------
# /predict ENDPOINT
# -----------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    print("\n========== NEW REQUEST ==========")
    print("FILES:", request.files)
    print("FORM:", request.form)  # FIXED

    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['file']

    try:
        # Read image
        img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Run model
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_index = int(np.argmax(output))

        if pred_index >= len(BREEDS):
            return jsonify({"error": "Prediction index out of range"}), 500

        predicted_breed = BREEDS[pred_index]
        confidence = float(output[pred_index])

        return jsonify({
            "breed": predicted_breed,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        import traceback
        print("\n🔥 SERVER CRASHED:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cow Breed API running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
