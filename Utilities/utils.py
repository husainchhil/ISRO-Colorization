import logging
import rasterio
import warnings
import rioxarray
import numpy as np
import xarray as xr
from tqdm import tqdm
import tensorflow as tf
from skimage import color
from tensorflow.keras.losses import Huber # type: ignore
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


loaded_model = tf.keras.models.load_model('Model/my_model.h5', compile=False)

# Recompile the model with valid loss and reduction
loaded_model.compile(optimizer='RMSprop',
                     loss=Huber(reduction='sum_over_batch_size'),
                     metrics=['accuracy'])

def get_data(catalog, bbox, time):
    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=time,
    )
    items = search.item_collection()
    return items


def build_and_split_images(items, patch_size=(256, 256)):
    def build_image(item):
        if "vv" in item.assets:
            vv = (
                rioxarray.open_rasterio(
                    item.assets["vv"].href, overview_level=2)
                .astype(float)
                .squeeze()
            )
        else:
            vv = xr.DataArray(np.zeros(patch_size, dtype=float))

        if "vh" in item.assets:
            vh = (
                rioxarray.open_rasterio(
                    item.assets["vh"].href, overview_level=2)
                .astype(float)
                .squeeze()
            )
        else:
            vh = xr.DataArray(np.zeros(patch_size, dtype=float))

        r = vv / 600
        g = vh / 270
        b = (vv / vh).where(vh > 0) / 9

        data = xr.concat([r, g, b], dim="band").clip(
            0, 1).where(lambda x: x > 0)

        if "sat:orbit_state" in item.properties and item.properties["sat:orbit_state"].lower() == "descending":
            data = np.flip(data, axis=(1, 2)).clip(0, 1)

        data = data.fillna(0)
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        image = (data).astype(np.float32).transpose(1, 2, 0)

        return image

    all_patches = []

    try:
        for item in tqdm(items):
            asset = item.assets.get('thumbnail')
            image_url = asset.href

            with rasterio.open(image_url) as src:
                image = src.read()

            h, w = image.shape[:2]
            patch_h, patch_w = patch_size

            built_image = build_image(item)

            for i in range(0, h, patch_h):
                for j in range(0, w, patch_w):
                    patch = built_image[i:i + patch_h, j:j + patch_w, :]
                    all_patches.append(patch)
        return all_patches
    except Exception as e:
        print(e)
        return None


def process_and_save_images(data, index, X_dim=256, Y_dim=256):
    filtered_data = []
    for i in range(len(data) - 1, -1, -1):
        try:
            if data[i].shape != (X_dim, Y_dim, 3):
                data.pop(i)
                print(f'Popped {i} from the list due to incorrect shape.')
            else:
                filtered_data.append(data[i])
        except AttributeError:
            data.pop(i)
            print(f'Popped {i} due to invalid data.')

    lab_images = [color.rgb2lab(
        image) for image in filtered_data if image is not None and image.shape[-1] == 3]
    lab_images = np.array(lab_images)

    return lab_images


def postprocess_image(tens_orig_l, out_ab, mode='bilinear'):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    logging.info(f"Postprocessing image with shape {tens_orig_l.shape} and {out_ab.shape}")
    HW_orig = tf.shape(tens_orig_l)[2:]
    HW = tf.shape(out_ab)[2:]

    # Resize if necessary
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = tf.image.resize(out_ab, size=HW_orig, method=mode)
    else:
        out_ab_orig = out_ab

    logging.info(f"Resized ab channels to shape {out_ab_orig.shape}")
    out_lab_orig = tf.concat([tens_orig_l, out_ab_orig], axis=1)

    logging.info(f"Concatenated L and ab channels to shape {out_lab_orig.shape}")
    out_lab_orig_np = out_lab_orig.numpy()
    out_rgb = color.lab2rgb(out_lab_orig_np[0, ...].transpose((1, 2, 0)))
    return out_rgb


def preprocess_image(image: np.ndarray) -> np.ndarray:
    logging.info(f"Checking color space of image with shape.")
    if image.ndim == 2:
        image = color.gray2rgb(image)
        print('The image is in grayscale and was converted to RGB')
    elif image.ndim == 3:
        print('The image is in RGB')
    elif image.shape[2] == 4:
        image = image[:, :, :3]
        print('The image is in RGBA and the alpha channel was removed')
    else:
        raise ValueError('The image is not in a recognized color space')
    image = tf.image.resize(image, (256, 256))
    logging.info(f"Resizing image to shape {image.shape}")

    image = tf.image.convert_image_dtype(image, tf.float32)
    logging.info(f"Converting image to float32 with shape {image.shape}")

    image = color.rgb2lab(image)
    logging.info(f"Converting image to LAB color space with shape {image.shape}")

    image_X = image[..., 0].reshape(1, 256, 256, 1) / 100
    logging.info(f"Extracting L channel with shape {image_X.shape}")

    return image_X


def colorize(image: np.ndarray) -> np.ndarray:
    try:
        logging.info(f"Preprocessing image with shape {image.shape}")
        X = preprocess_image(image)

        logging.info(f"Predicting ab channels for image with shape {X.shape}")
        predicted_ab = loaded_model.predict(X/100)

        logging.info(f"Postprocessing image with shape {predicted_ab.shape}")
        tens_orig_l = tf.convert_to_tensor(X.reshape(
            # L channel scaled to [0, 100]
            1, 1, 256, 256), dtype=tf.float32)

        # Convert predicted_ab to TensorFlow Tensor and scale it back to [-128, 128]
        out_ab = tf.convert_to_tensor(predicted_ab, dtype=tf.float32)
        out_ab = tf.transpose(out_ab, perm=[0, 3, 1, 2])  # Reshape from NHWC to NCHW
        out_ab = out_ab * 128  # Scale the predicted ab channels

        # Assuming postprocess_tens is a function that converts L and ab channels to RGB
        predicted_rgb = postprocess_image(tens_orig_l, out_ab)

        return predicted_rgb
    except Exception as e:
        print(e)
        return None