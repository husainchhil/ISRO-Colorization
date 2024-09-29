from fastapi import FastAPI, HTTPException, Form, UploadFile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Utilities.utils import preprocess_image, postprocess_image

app = FastAPI(
    title="SAR Image Colorization",
    description="This is a simple API to colorize SAR images.",
    version="0.1",
)

@app.post("/colorize/")
async def colorize_image(file: UploadFile = Form(...)):
    try:
        image = plt.imread(file.file)
        print(image.shape)
        # model = tf.keras.models.load_model("Model/my_model.h5")
        # processed_image = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
