from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import Huber
from PIL import Image
import io
from Utilities.utils import preprocess_image, postprocess_image

app = FastAPI(
    title="SAR Image Colorization",
    description="This is a simple API to colorize SAR images.",
    version="0.1",
)
# Load the model without compiling
loaded_model = tf.keras.models.load_model('Model/my_model.h5', compile=False)

# Recompile the model with valid loss and reduction
loaded_model.compile(optimizer='RMSprop',
                     loss=Huber(reduction='sum_over_batch_size'),
                     metrics=['accuracy'])

@app.post("/colorize/")
async def colorize_image(file: UploadFile = File(..., description="Upload a grayscale (Grayscale, RGB, or RGBA) image to colorize.")):
    try:
                # Read the image bytes
        image_data = file.file.read()

        # Convert bytes to a PIL image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Preprocess the image
        X = preprocess_image(image_np)

        predicted_ab = loaded_model.predict(X/100)

        # Preprocessing step (assuming L channel is in the range [0, 1])
        tens_orig_l = tf.convert_to_tensor(X.reshape(
            # L channel scaled to [0, 100]
            1, 1, 256, 256), dtype=tf.float32)

        # Convert predicted_ab to TensorFlow Tensor and scale it back to [-128, 128]
        out_ab = tf.convert_to_tensor(predicted_ab, dtype=tf.float32)
        out_ab = tf.transpose(out_ab, perm=[0, 3, 1, 2])  # Reshape from NHWC to NCHW
        out_ab = out_ab * 128  # Scale the predicted ab channels

        # Assuming postprocess_tens is a function that converts L and ab channels to RGB
        predicted_rgb = postprocess_image(tens_orig_l, out_ab)

        # Visualize the image
        plt.imsave("output.png", predicted_rgb)

        # Check the range of the predicted RGB image
        print(f"RGB Min: {tf.reduce_min(predicted_rgb).numpy()}, RGB Max: {tf.reduce_max(predicted_rgb).numpy()}")
        return FileResponse("output.png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
