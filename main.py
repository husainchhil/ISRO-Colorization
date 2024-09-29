import sys
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import Huber # type: ignore
import logging
from PIL import Image
import io
from Utilities.utils import preprocess_image, postprocess_image, colorize

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(
    title="SAR Image Colorization",
    description="This is a simple API to colorize SAR images.",
    version="0.1",
)

@app.post("/colorize/")
async def colorize_image(file: UploadFile = File(..., description="Upload a grayscale (Grayscale, RGB, or RGBA) image to colorize.")):
    try:
        image_data = file.file.read()
        logging.info(f"Received image with size {len(image_data)} bytes.")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logging.info(f"Colorizing image...")
        predicted_rgb = colorize(np.array(image))

        plt.imsave("output.png", predicted_rgb)
        logging.info("Image colorized successfully.")

        return FileResponse("output.png")
    except Exception as e:
        lineno = sys.exc_info()[-1].tb_lineno
        logging.error(f"Error at line {lineno}: \n\n{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
