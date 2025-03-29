import os
import numpy as np
import cv2
import tifffile as tiff
import logging

# Setup logging
logging.basicConfig(filename='inpaint_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def inpaint_darkest_region_preserving_tissue(image, inpaint_method=cv2.INPAINT_NS):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[tissue_mask == 255] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=15, flags=inpaint_method)
    return inpainted, mask

# Define base input/output folders
base_input_dir = "/users/ledunuri/bat_guts"
base_output_dir = "/users/ledunuri/cleaned_images"

print("📂 Base input directory:", base_input_dir)
logging.info(f"Base input directory: {base_input_dir}")

if not os.path.exists(base_input_dir):
    print("❌ ERROR: Base input folder does not exist.")
    logging.error("Base input folder does not exist.")
    exit()

subfolders = [f for f in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, f))]
print("🔍 Subfolders found:", subfolders)
logging.info(f"Subfolders found: {subfolders}")

# Loop through all input subfolders
for folder in subfolders:
    folder_path = os.path.join(base_input_dir, folder)
    output_folder = os.path.join(base_output_dir, folder)
    os.makedirs(output_folder, exist_ok=True)

    print(f"📁 Entering folder: {folder}")
    logging.info(f"Processing folder: {folder}")

    # Go one level deeper
    subsubfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    tiff_files = []
    for subsub in subsubfolders:
        for f in os.listdir(subsub):
            if f.lower().endswith((".tif", ".tiff")):
                tiff_files.append(os.path.join(subsub, f))

    print(f"🖼 Found {len(tiff_files)} TIFF files in {folder}")
    logging.info(f"Found {len(tiff_files)} TIFF files in {folder}")

    for input_path in tiff_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_folder, f"inpainted_{filename}")

        try:
            image = tiff.imread(input_path)
            inpainted_image, mask = inpaint_darkest_region_preserving_tissue(image)
            tiff.imwrite(output_path, inpainted_image)

            print(f"✅ Processed: {input_path}")
            logging.info(f"Processed: {input_path} → {output_path}")

        except Exception as e:
            print(f"❌ Failed on {input_path}: {e}")
            logging.error(f"Failed on {input_path}: {e}")


print("🎉 All folders processed successfully!")
logging.info("All folders processed successfully.")
