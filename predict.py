import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- DEBUG: SHOW FILES ----------------
print("Current directory:", os.getcwd())
print("Files in folder:", os.listdir())

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"   # YOUR TFLITE MODEL NAME
IMAGE_SIZE = (224, 224)

# ---------------- CLASS NAMES (YOUR TRAINING ORDER) ----------------
CLASS_NAMES = [
    "Alambadi",
    "Amritmahal",
    "Ayrshire",
    "Banni",
    "Bargur",
    "Bhadawari",
    "Brown_Swiss",
    "Dangi",
    "Deoni",
    "Gir",
    "Guernsey",
    "Hallikar",
    "Hariana",
    "Holstein_Friesian",
    "Jaffrabadi",
    "Jersey",
    "Kangayam",
    "Kankrej",
    "Kasargod",
    "Kenkatha",
    "Kherigarh",
    "Khillari",
    "Krishna_Valley",
    "Malnad_gidda",
    "Mehsana",
    "Murrah",
    "Nagori",
    "Nagpuri",
    "Nili_Ravi",
    "Nimari",
    "Ongole",
    "Pulikulam",
    "Rathi",
    "Red_Dane",
    "Red_Sindhi",
    "Sahiwal",
    "Surti",
    "Tharparkar",
    "Toda",
    "Umblachery",
    "Vechur"
]

# ---------------- LOAD TFLITE MODEL ----------------
print("\nLoading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✔ Model loaded!\n")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)

    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image_path):
    img = preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return CLASS_NAMES[idx], confidence

# ---------------- TEST IMAGE PATH ----------------
TEST_IMAGE = r"C:\Users\karthikarthika\Downloads\images.jpg"

breed, conf = predict_image(TEST_IMAGE)

print("========= PREDICTION RESULT =========")
print("Breed:", breed)
print("Confidence:", conf)
print("=====================================")
