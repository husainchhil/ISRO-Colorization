"""
This module provides an API for colorizing SAR images using FastAPI.
Functions:
    colorize_image(file: UploadFile) -> FileResponse:
        Endpoint to colorize a grayscale (Grayscale, RGB, or RGBA) image.
        Accepts an image file upload and returns the colorized image.
Modules:
    io: Provides the core tools for working with streams.
    sys: Provides access to some variables used or maintained by the interpreter.
    logging: Provides a way to configure and use a flexible event logging system.
    numpy: Provides support for large, multi-dimensional arrays and matrices.
    PIL: Python Imaging Library, adds image processing capabilities.
    matplotlib.pyplot: Provides a MATLAB-like interface for plotting.
    fastapi.responses: Provides various response classes for FastAPI.
    fastapi: A modern, fast (high-performance) web framework for building APIs with Python 3.6+.
    Utilities.utils: Custom utility module containing the colorize function.
Logging:
    Configured to log INFO level messages with a specific format.
FastAPI App:
    Configured with title, description, and version.
Endpoints:
    /colorize/:
        POST:
            Description: Upload a grayscale (Grayscale, RGB, or RGBA) image to colorize.
            Parameters:
                file: UploadFile - The image file to be colorized.
            Returns:
                FileResponse - The colorized image file.
            Raises:
                HTTPException - If an error occurs during processing.
"""

import io
import sys
import uvicorn
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File
from Utilities.utils import colorize

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
        image_shape = Image.open(io.BytesIO(image_data)).size[::-1]
        logging.info(f"Received image with size {len(image_data)} bytes.")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        logging.info(f"Colorizing image...")
        predicted_rgb = colorize(np.array(image), image_shape)

        plt.imsave("output.png", predicted_rgb)
        logging.info("Image colorized successfully.")

        return FileResponse("output.png")
    except Exception as e:
        lineno = sys.exc_info()[-1].tb_lineno
        logging.error(f"Error at line {lineno}: \n\n{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)