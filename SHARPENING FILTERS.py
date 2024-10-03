import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("nature.jpg")

# Convert the image to RGB color space
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Apply the Laplacian operator
laplacian = cv2.Laplacian(image2, cv2.CV_64F)  # Use CV_64F for better precision

# Convert the Laplacian image back to uint8 for display
laplacian = cv2.convertScaleAbs(laplacian)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

# Display the Laplacian filtered image
plt.subplot(1, 2, 2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")

plt.show()