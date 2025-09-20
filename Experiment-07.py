import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
path = "C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png"   # Replace with your image path
img = cv2.imread(path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------- 1. Box Filter (Average Filter) -----------------
# ksize = (kernel size)
box = cv2.blur(img_rgb, ksize=(5,5))

# ----------------- 2. Median Filter -----------------
median = cv2.medianBlur(img_rgb, ksize=5)  # ksize must be odd

# ----------------- 3. Max Filter -----------------
# Using dilation as Max filter
kernel = np.ones((5,5), np.uint8)
max_filter = cv2.dilate(img_rgb, kernel)

# ----------------- 4. Min Filter -----------------
# Using erosion as Min filter
min_filter = cv2.erode(img_rgb, kernel)

# ----------------- 5. Weighted Average Filter (Gaussian Blur) -----------------
weighted = cv2.GaussianBlur(img_rgb, ksize=(5,5), sigmaX=1.5)

# ----------------- Display Results -----------------
titles = ['Original', 'Box Filter', 'Median Filter', 'Max Filter', 'Min Filter', 'Weighted Average']
images = [img_rgb, box, median, max_filter, min_filter, weighted]

plt.figure(figsize=(15,10))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()
