import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- LOAD MODEL --------------------
print("Loading model...")
model = tf.keras.models.load_model("Best_Cattle_Breed.h5")
print("Model loaded successfully!")

# ------------------ BREED CLASS NAMES ------------------
CLASS_NAMES = [
    "Umblachery", "Tharparkar", "Toda", "Sahiwal", "Surti", "Red_Dane",
    "Rathi", "Pulikulam", "Ongole", "Nimari", "Nagpuri", "Nili_Ravi",
    "Nagori", "Murrah", "Mehsana", "Malnad_gidda", "Krishna_Valley",
    "Khillari", "Kasargod", "Kenkatha", "Kherigarh", "Kankrej",
    "Kangayam", "Jaffrabadi", "Jersey", "Holstein_Friesian", "Hariana",
    "Hallikar", "Guernsey", "Gir", "Deoni", "Dangi", "Bhadawari",
    "Brown_Swiss", "Bargur", "Banni", "Ayrshire", "Amritmahal",
    "Alambadi"
]

IMAGE_SIZE = (224, 224)

# -------------------- PREPROCESS IMAGE --------------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)

    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------- PREDICT FUNCTION --------------------
def predict_cow(img):
    arr = preprocess(img)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    
    breed = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

    return {
        "breed": breed,
        "confidence": round(confidence, 4)
    }

# -------------------- GRADIO INTERFACE --------------------
demo = gr.Interface(
    fn=predict_cow,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Indian Cow Breed Classifier",
    description="Upload a cow image to get breed prediction and confidence.",
)

demo.launch()
