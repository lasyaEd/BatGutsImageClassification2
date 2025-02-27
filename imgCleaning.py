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
    '''''
    Why thresholding?
        It helps isolate dark objects (like the scale bar & text).
    What does cv2.THRESH_BINARY_INV do?
        Converts all dark pixels (intensity ≤ 1) to white (255).
        Converts all bright pixels to black (0).
    This creates a binary mask, where:
        White (255) = Areas to be removed (scale bar & text).
        Black (0) = Areas to be preserved (background & tissue).
    '''''
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)  

    # **NEW STEP: Detect tissue region (using Otsu’s thresholding)**
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    '''''
    What is Otsu’s thresholding?

    Automatically finds an optimal threshold to separate foreground (tissue) from background.
    This ensures that all tissue is detected without manually setting a threshold.
    What does this mask contain?

    White (255) = Tissue areas.
    Black (0) = Background & unwanted areas.
    '''''

    # **Exclude tissue regions from the inpainting mask**
    mask[tissue_mask == 255] = 0  # Remove any part of the mask that overlaps with tissue

    ''' 
    This modifies the mask so that any pixels inside the tissue are ignored.
    Why?
    The scale bar & text are dark, but some tissue areas might also be dark.
    Without this step, the script could accidentally inpaint inside the tissue.
    How?
    Wherever the tissue_mask == 255 (i.e., inside the tissue), the mask is set to 0 (excluded from inpainting).
    '''

    # **Expand the mask slightly to remove thin white edges**
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)  # Expand dark region coverage

    ''''
    Why?
    The detected mask may have thin white edges around the scale bar or text.
    Dilation expands the mask slightly to fully cover the unwanted regions.
    How?
    Uses a 5x5 kernel (structuring element).
    2 iterations to grow the mask slightly.
    '''

    # Inpaint only the detected dark region (scale bar + text)
    inpainted = cv2.inpaint(image, mask, inpaintRadius=15, flags=inpaint_method)  # Larger radius for smooth blending

    '''

    What is inpainting?
    A technique that fills missing regions in an image using surrounding pixels.
    Parameters:
    mask: Defines the regions to be removed (scale bar & text).
    inpaintRadius=15: Controls how far pixels are sampled for filling. Higher values blend better.
    flags=cv2.INPAINT_NS: Uses Navier-Stokes inpainting, which is good for smooth regions.

    '''

    return inpainted, mask

# Load the TIFF image (Change path to your file)
tiff_path = "/Users/lasyaedunuri/Documents/InternshipYo/BatGutsImageClassification2/T2023-26L_Anoura_geoffroyi_LY24-1-7_nectar/T2023-26L_Anoura_geoffroyi_LY24-1-7A_nectar_stomach_AB-PAS/T2023-26L_Anoura_geoffroyi_LY24-1-7A_nectar_stomach_AB-PAS_LY24-1-7A AB-PAS 1-2.tif"
image = tiff.imread(tiff_path)

# Inpaint using Navier-Stokes (`cv2.INPAINT_NS`) while preserving tissue
inpainted_image, mask = inpaint_darkest_region_preserving_tissue(image, inpaint_method=cv2.INPAINT_NS)

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

# Optional: Save the inpainted image
tiff.imwrite("inpainted_image_tissue_preserved.tif", inpainted_image)
