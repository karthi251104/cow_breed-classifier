from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.tflite"   # your TFLite model file
IMAGE_SIZE = (224, 224)

# ---------------- CLASS NAMES ----------------
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
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✔ TFLite model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
