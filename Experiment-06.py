import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "C:/Users/luthr/OneDrive/Pictures/Screenshots 1/Screenshot 2025-09-19 121038.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# ---------------- Sobel Operator ----------------
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# ---------------- Prewitt Operator ----------------
prewitt_kernel_x = np.array([[ -1, 0, 1], 
                             [ -1, 0, 1], 
                             [ -1, 0, 1]])
prewitt_kernel_y = np.array([[ -1, -1, -1], 
                             [  0,  0,  0], 
                             [  1,  1,  1]])
prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))

# ---------------- Roberts Cross Operator ----------------
roberts_cross_v = np.array([[1, 0], [0, -1]])
roberts_cross_h = np.array([[0, 1], [-1, 0]])
roberts_v = cv2.filter2D(img, -1, roberts_cross_v)
roberts_h = cv2.filter2D(img, -1, roberts_cross_h)
roberts = cv2.magnitude(roberts_v.astype(float), roberts_h.astype(float))

# ---------------- Canny Edge Detection ----------------
# threshold1 and threshold2 can be tuned for better edges
canny = cv2.Canny(img, threshold1=100, threshold2=200)

# ---------------- Display results ----------------
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Sobel Operator")
plt.imshow(sobel, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Prewitt Operator")
plt.imshow(prewitt, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Roberts Cross Operator")
plt.imshow(roberts, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Canny Edge Detection")
plt.imshow(canny, cmap='gray')
plt.axis("off")

plt.show()
