
import numpy as np
from PIL import Image

try:
    # Try tflite-runtime (lightweight)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        # Try TensorFlow's TFLite
        from tensorflow.lite import Interpreter
    except ImportError:
        # Last fallback: use tflite-support
        from tflite_support import flatbuffers
        from tflite_support import metadata
        raise ImportError("No valid TFLite interpreter found. Please ensure tflite-runtime or tensorflow is installed.")



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
