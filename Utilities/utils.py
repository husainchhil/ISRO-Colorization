import rasterio
import warnings
import rioxarray
import numpy as np
import xarray as xr
from tqdm import tqdm
import tensorflow as tf
from skimage import color
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


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

    HW_orig = tf.shape(tens_orig_l)[2:]
    HW = tf.shape(out_ab)[2:]

    # Resize if necessary
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = tf.image.resize(out_ab, size=HW_orig, method=mode)
    else:
        out_ab_orig = out_ab

    # Concatenate L and ab channels to form LAB image
    out_lab_orig = tf.concat([tens_orig_l, out_ab_orig], axis=1)

    # Convert LAB to RGB
    out_lab_orig_np = out_lab_orig.numpy()
    out_rgb = color.lab2rgb(out_lab_orig_np[0, ...].transpose((1, 2, 0)))
    return out_rgb


def preprocess_image(image: np.ndarray) -> np.ndarray:
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
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = color.rgb2lab(image)
    image_X = image[..., 0].reshape(1, 256, 256, 1) / 100
    return image_X
