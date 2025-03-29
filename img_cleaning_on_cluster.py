import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

def inpaint_darkest_region_preserving_tissue(image, inpaint_method=cv2.INPAINT_NS):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[tissue_mask == 255] = 0
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=15, flags=inpaint_method)
    return inpainted, mask

# Base input and output directories on the cluster
base_input_dir = "/users/ledunuri"
base_output_dir = "/users/ledunuri/cleaned_images"

# Loop through all folders in the base input directory
for folder in os.listdir(base_input_dir):
    folder_path = os.path.join(base_input_dir, folder)
    if os.path.isdir(folder_path):
        # Create corresponding output folder
        output_folder = os.path.join(base_output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith(".tiff"):
                input_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_folder, f"inpainted_{filename}")

                # Load TIFF
                image = tiff.imread(input_path)

                # Inpaint
                inpainted_image, mask = inpaint_darkest_region_preserving_tissue(image)

                # Save result
                tiff.imwrite(output_path, inpainted_image)
                print(f"✅ Processed: {input_path} → {output_path}")

print("🎉 All folders processed successfully!")
