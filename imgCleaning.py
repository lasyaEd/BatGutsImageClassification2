import os
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

def inpaint_darkest_region_preserving_tissue(image, inpaint_method=cv2.INPAINT_NS):
    """
    Detects and inpaints the darkest region in the image (scale bar + text),
    while preserving the tissue.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the darkest region using thresholding (for scale bar + text)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Detect tissue region using Otsu’s thresholding
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Exclude tissue regions from the inpainting mask
    mask[tissue_mask == 255] = 0  

    # Expand the mask slightly to remove thin white edges
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Inpaint only the detected dark region (scale bar + text)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=15, flags=inpaint_method)

    return inpainted, mask

# Define input folder containing TIFF images
input_folder = "/Users/lasyaedunuri/Documents/InternshipYo/BatGutsImageClassification2/images/insect"

# Define output folder for inpainted images
output_folder = "/Users/lasyaedunuri/Documents/InternshipYo/Inpainted_Images/insect"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn’t exist

# Loop through all TIFF files in the directory and process them
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):  # Process only TIFF files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"inpainted_{filename}")

        # Load the TIFF image
        image = tiff.imread(input_path)

        # Inpaint the image
        inpainted_image, mask = inpaint_darkest_region_preserving_tissue(image)

        # Save the inpainted image
        tiff.imwrite(output_path, inpainted_image)

        # Display results
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("Final Mask (Scale Bar + Text Only)")
        ax[1].axis("off")

        ax[2].imshow(inpainted_image)
        ax[2].set_title("Inpainted Image (Tissue Untouched)")
        ax[2].axis("off")

        plt.show()

        print(f"Processed: {filename} → Saved as: {output_path}")

print("✅ Batch processing complete!")
