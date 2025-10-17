import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
path = "C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-10-17 115213.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Create kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)

# Perform Opening: Erosion followed by Dilation
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Perform Closing: Dilation followed by Erosion
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Display results
titles = ['Original Image', 'Opening', 'Closing']
images = [img, opening, closing]

plt.figure(figsize=(10, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
