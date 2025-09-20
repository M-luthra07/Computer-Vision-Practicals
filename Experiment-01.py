
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV

# Step 1: Read the image
# Replace 'your_image.jpg' with your actual image path
img = cv2.imread("C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png")

# OpenCV loads in BGR, convert to RGB for visualization
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Convert image into array
img_array = np.array(img_rgb)

# Print shape and data type
print("Array shape:", img_array.shape)   # (height, width, channels)
print("Data type:", img_array.dtype)

# Step 3: Visualize original image
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# Step 4: Plot grayscale version
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# Step 5: Display histogram of pixel values
plt.hist(gray_img.ravel(), bins=256, color='black')
plt.title("Pixel Intensity Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

