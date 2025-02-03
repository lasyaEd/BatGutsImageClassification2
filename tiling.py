import os
import numpy as np
import cv2
from PIL import Image

def load_tiff_as_numpy(image_path, channels=3):
    """
    Loads a TIFF image and converts it into a NumPy array.

    Parameters:
    - image_path: Path to the TIFF image.
    - channels: Number of channels to load (1 = Grayscale, 3 = RGB).

    Returns:
    - NumPy array representing the image.
    """
    image = Image.open(image_path)

    if channels == 1:
        image = image.convert("L")  # Convert to grayscale
    else:
        image = image.convert("RGB")  # Ensure RGB mode

    image_array = np.array(image)  # Convert to NumPy array
    print(f"Loaded image shape: {image_array.shape}")  # Debugging output
    return image_array


def tile_numpy_image(image_array, output_folder, image_name, tile_size=(512, 512), overlap=0.2):
    """
    Splits a NumPy image into smaller overlapping tiles and saves them with a matching name.

    Parameters:
    - image_array: NumPy array of the image (Height x Width x Channels).
    - output_folder: Folder where tiles will be saved.
    - image_name: Base name of the original image file.
    - tile_size: Tuple (tile_width, tile_height) for each tile (default: 512x512).
    - overlap: Fraction of overlap between tiles (default: 20%).

    Saves:
    - Tiles as image files in the specified folder, named as "image_name_01.png", "image_name_02.png", etc.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    img_h, img_w = image_array.shape[:2]
    tile_w, tile_h = tile_size

    # Calculate stride (step size) based on overlap
    stride_w = int(tile_w * (1 - overlap))
    stride_h = int(tile_h * (1 - overlap))

    tile_count = 1

    # Slide over the image and extract patches
    for y in range(0, img_h - tile_h + 1, stride_h):
        for x in range(0, img_w - tile_w + 1, stride_w):
            tile = image_array[y:y + tile_h, x:x + tile_w]

            # Construct the new tile name
            tile_filename = os.path.join(output_folder, f"{image_name}_{tile_count:02d}.png")

            # Save tile as image
            if len(tile.shape) == 2:  # Grayscale (1 channel)
                cv2.imwrite(tile_filename, tile)
            else:  # RGB (3 channels)
                cv2.imwrite(tile_filename, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
            
            tile_count += 1

    print(f"Tiling complete. {tile_count - 1} tiles saved in {output_folder}")


# Example Usage:
tiff_image_path = "/Users/lasyaedunuri/Documents/InternshipYo/BatGutsImageClassification2/T2023-26L_Anoura_geoffroyi_LY24-1-7_nectar/T2023-26L_Anoura_geoffroyi_LY24-1-7B_nectar_proximal_AB-PAS/T2023-26L_Anoura_geoffroyi_LY24-1-7B_nectar_proximal_AB-PAS_LY24-1-7B AB-PAS 1-1.tif"
output_folder = "tiled_images"
image_name = os.path.splitext(os.path.basename(tiff_image_path))[0]  # Extract filename without extension

# Load TIFF as NumPy array
image_array = load_tiff_as_numpy(tiff_image_path, channels=3)  # Change to 1 for grayscale

# Convert NumPy array into tiles and save them
tile_numpy_image(image_array, output_folder, image_name, tile_size=(512, 512), overlap=0.2)
