import cv2
import numpy as np
import matplotlib.pyplot as plt
path = "C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("⚠️ Image not found. Please check the file path.")
    exit()
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(erosion, cmap='gray')
plt.title("After Erosion")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(dilation, cmap='gray')
plt.title("After Dilation")
plt.axis('off')

plt.show()
