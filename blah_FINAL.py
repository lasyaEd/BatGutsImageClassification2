import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

def inpaint_darkest_region(image, inpaint_method=cv2.INPAINT_NS):
    """
    Detects and inpaints the darkest region in the image, ensuring proper blending.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect the darkest region using thresholding
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)  

    # **NEW FIX: Expand the mask slightly to remove thin white edges**
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)  # Expand dark region coverage

    # Inpaint the detected dark region using the Navier-Stokes method
    inpainted = cv2.inpaint(image, mask, inpaintRadius=15, flags=inpaint_method)  # Larger radius for smooth blending

    return inpainted, mask

# Load the TIFF image (Change path to your file)
tiff_path = "/Users/lasyaedunuri/Documents/InternshipYo/BatGutsImageClassification2/T2023-26L_Anoura_geoffroyi_LY24-1-7_nectar/T2023-26L_Anoura_geoffroyi_LY24-1-7C_nectar_middle_AB-PAS/T2023-26L_Anoura_geoffroyi_LY24-1-7C_nectar_middle_AB-PAS_LY24-1-7C AB-PAS 1-2.tif"
image = tiff.imread(tiff_path)

# Inpaint using Navier-Stokes (`cv2.INPAINT_NS`)
inpainted_image, mask = inpaint_darkest_region(image, inpaint_method=cv2.INPAINT_NS)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Expanded Mask")
ax[1].axis("off")

ax[2].imshow(inpainted_image)
ax[2].set_title("Inpainted Image (Navier-Stokes, Fixed)")
ax[2].axis("off")

plt.show()

# Optional: Save the inpainted image
tiff.imwrite("inpainted_image_fixed.tif", inpainted_image)
