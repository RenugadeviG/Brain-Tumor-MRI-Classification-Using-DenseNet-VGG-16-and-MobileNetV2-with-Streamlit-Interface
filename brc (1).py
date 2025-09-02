import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

MODEL_PATH = "model.tflite"
CLASS_MAP_PATH = "class_map.json"

# Load class map
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)
idx_to_class = {int(k): v for k, v in class_map.items()}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Upload and predict
uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    x = np.array(img.resize((224,224)), dtype=np.float32)/255.0
    x = np.expand_dims(x, axis=0)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred_probs = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(pred_probs)
    st.write(f"Prediction: {idx_to_class[pred_idx]} ({pred_probs[pred_idx]*100:.2f}%)")

