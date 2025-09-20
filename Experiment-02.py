# Import libraries
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# File path (replace with your image)
path = "C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png"

# ----------- 1. Using OpenCV -------------
img_cv = cv2.imread(path)             # Read (BGR format)
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

print("OpenCV Image Shape:", img_cv.shape)

plt.subplot(1,3,1)
plt.imshow(img_cv_rgb)
plt.title("OpenCV")
plt.axis("off")

# ----------- 2. Using Matplotlib -------------
img_plt = plt.imread(path)            # Directly reads in RGB format
print("Matplotlib Image Shape:", img_plt.shape)

plt.subplot(1,3,2)
plt.imshow(img_plt)
plt.title("Matplotlib")
plt.axis("off")

# ----------- 3. Using PIL (Pillow) -------------
img_pil = Image.open(path)            # Open with PIL
img_pil_array = np.array(img_pil)     # Convert to numpy array
print("PIL Image Shape:", img_pil_array.shape)

plt.subplot(1,3,3)
plt.imshow(img_pil)
plt.title("PIL")
plt.axis("off")

plt.show()
