"""
This Streamlit application allows users to upload grayscale images and colorize them using a deep learning model.
Modules:
    io: Core tools for working with streams.
    sys: System-specific parameters and functions.
    logging: Provides a flexible framework for emitting log messages from Python programs.
    numpy: Fundamental package for scientific computing with Python.
    PIL: Python Imaging Library for opening, manipulating, and saving many different image file formats.
    streamlit: Framework for creating web apps.
    matplotlib.pyplot: State-based interface to matplotlib for plotting.
    Utilities.utils: Custom module containing the colorize function.
Functions:
    colorize(image: np.ndarray, image_shape: tuple) -> np.ndarray:
        Colorizes the given grayscale image using a deep learning model.
Streamlit Components:
    st.set_page_config: Sets the configuration of the Streamlit app.
    st.title: Displays the main title of the app.
    st.subheader: Displays a subheader in the app.
    st.info: Displays an informational message.
    st.divider: Displays a horizontal divider.
    st.file_uploader: Allows users to upload an image file.
    st.button: Creates a button that users can click.
    st.columns: Creates columns for layout purposes.
    st.image: Displays an image.
    st.download_button: Creates a button to download files.
    st.info: Displays a info notification.
    st.error: Displays an error message.
Logging:
    Configured to log messages with level INFO and above.
    Logs the size of the uploaded image.
    Logs the start and success of the colorization process.
    Logs any errors that occur during the colorization process.
Usage:
    1. Upload a grayscale image using the file uploader.
    2. Click the "Colorize Image" button to colorize the uploaded image.
    3. View the original and colorized images side by side.
    4. Download the colorized image using the download button.
"""
import io
import sys
import logging
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from Utilities.utils import colorize


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="SAR Image Colorization", page_icon=Image.open('Assets/image.png'), layout="centered")

st.title("🌈SAR Image Colorization")

st.subheader("Colorize SAR images using a Deep Learning Model trained on SAR (Satellites for the dumb) Images.🚀")
st.info("The model is specifically trained to colorize SAR images, so make sure to upload a SAR image for the best results.")

st.divider()

image = st.file_uploader("Upload a grayscale (Grayscale, RGB, or RGBA) image to colorize.", type=["jpg", "jpeg", "png"])

if image is not None:
    colorize_button = st.button("Colorize Image", use_container_width=True)
    if colorize_button:
        try:
            image_data = image.read()
            image_shape = Image.open(io.BytesIO(image_data)).size[::-1]
            image_name = image.name

            logging.info(f"Received image with size {len(image_data)} bytes.")

            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            logging.info(f"Colorizing image...")
            predicted_rgb = colorize(np.array(image), image_shape)

            plt.imsave("output.png", predicted_rgb)
            logging.info("Image colorized successfully.")
            
            gray_col, color_col = st.columns(2)
            gray_col.image(image, caption="Grayscale Image", use_column_width=True)
            color_col.image(predicted_rgb, caption="Colorized Image", use_column_width=True)
            with open("output.png", "rb") as file:
                st.download_button(label="Download Colorized Image", data=file, file_name=f'{image_name.split(".")[0]}_colorized.png', mime="image/png", use_container_width=True)

        except Exception as e:
            lineno = sys.exc_info()[-1].tb_lineno
            logging.error(f"Error at line {lineno}: \n\n{str(e)}")
            st.error("An error occurred while colorizing the image. Please try again.")
    else:
        st.toast("Click the 'Colorize Image' button to colorize the uploaded image.")
else:
    st.toast("Upload an image to get started.")