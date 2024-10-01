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