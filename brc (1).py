# ❌ Remove this
# import tensorflow as tf

# ✅ Use TensorFlow Lite runtime
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image

# Load TFLite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to run prediction
def predict(img):
    # Preprocess image
    img = img.resize((224, 224))   # adjust to your model input size
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
