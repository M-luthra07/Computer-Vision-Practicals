
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png')  # Replace with your image paths
img2 = cv2.imread("C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-20 123546.png")

# Convert to RGB for visualization
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Resize second image to match first image size (if needed)
img2_rgb = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))

# ------------------- 1. Arithmetic Operations -------------------
# Addition
add_img = cv2.add(img1_rgb, img2_rgb)

# Subtraction
sub_img = cv2.subtract(img1_rgb, img2_rgb)

# Multiplication
mul_img = cv2.multiply(img1_rgb, 0.5)  # scale factor 0.5

# Division
div_img = cv2.divide(img1_rgb, 2)  # divide by scalar

# ------------------- 2. Logical Operations -------------------
# Convert to grayscale for bitwise operations
gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

# Threshold to make binary images
_, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
_, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

# Bitwise AND
and_img = cv2.bitwise_and(binary1, binary2)

# Bitwise OR
or_img = cv2.bitwise_or(binary1, binary2)

# Bitwise XOR
xor_img = cv2.bitwise_xor(binary1, binary2)

# Bitwise NOT (invert first image)
not_img = cv2.bitwise_not(binary1)

# ------------------- 3. Visualization -------------------
titles = ['Original1', 'Original2', 'Addition', 'Subtraction', 'Multiplication', 'Division',
          'AND', 'OR', 'XOR', 'NOT']
images = [img1_rgb, img2_rgb, add_img, sub_img, mul_img, div_img, 
          and_img, or_img, xor_img, not_img]

plt.figure(figsize=(18,12))
for i in range(len(images)):
    plt.subplot(2,5,i+1)
    if i < 6:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='gray')  # logical operations in gray
    plt.title(titles[i])
    plt.axis('off')
plt.show()
