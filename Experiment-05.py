
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# ----------------- Load Images -----------------
# Source image to enhance
source_path = 'C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png'        # Replace with your image path
# Reference image for histogram matching
reference_path = 'C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-20 123546.png'  # Replace with your image path

# Load images
source_img = cv2.imread(source_path)
reference_img = cv2.imread(reference_path)

# Convert BGR → RGB for visualization
source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

# ----------------- Grayscale Version -----------------
source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)

# Histogram Equalization
equalized_gray = cv2.equalizeHist(source_gray)

# Histogram Matching (grayscale)
matched_gray = exposure.match_histograms(source_gray, reference_gray)

# ----------------- Color Version (Channel-wise) -----------------
matched_color = exposure.match_histograms(source_rgb, reference_rgb)

# ----------------- Visualization -----------------
# Grayscale images
titles_gray = ['Original Gray', 'Equalized Gray', 'Reference Gray', 'Matched Gray']
images_gray = [source_gray, equalized_gray, reference_gray, matched_gray]

plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images_gray[i], cmap='gray')
    plt.title(titles_gray[i])
    plt.axis('off')
plt.show()

# Color images
titles_color = ['Original Color', 'Reference Color', 'Histogram Matched Color']
images_color = [source_rgb, reference_rgb, matched_color]

plt.figure(figsize=(12,6))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images_color[i])
    plt.title(titles_color[i])
    plt.axis('off')
plt.show()

# ----------------- Histograms (Optional) -----------------
plt.figure(figsize=(12,6))
for i, img in enumerate([source_gray, equalized_gray, matched_gray]):
    plt.subplot(1,3,i+1)
    plt.hist(img.ravel(), bins=256, color='black')
    plt.title(titles_gray[i] + " Histogram")
plt.show()
