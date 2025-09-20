import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png', cv2.IMREAD_GRAYSCALE)

# 1. Image Negative
negative_img = 255 - img

# 2. Brightness Adjustment
brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=50)  # add 50 to each pixel

# 3. Contrast Stretching
contrast_img = cv2.convertScaleAbs(img, alpha=2, beta=0)  # double the intensity

# 4. Thresholding
_, threshold_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# 5. Log Transformation
c = 255 / np.log(1 + np.max(img))
log_img = c * np.log(1 + img)
log_img = np.array(log_img, dtype=np.uint8)

# 6. Power-Law (Gamma) Transformation
gamma = 0.5
gamma_img = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

# Visualization
titles = ['Original', 'Negative', 'Brightness +50', 'Contrast x2', 'Thresholding', 'Log Transform', 'Gamma Transform']
images = [img, negative_img, brightness_img, contrast_img, threshold_img, log_img, gamma_img]

plt.figure(figsize=(18,10))
for i in range(len(images)):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()
