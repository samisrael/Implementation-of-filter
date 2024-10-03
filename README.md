# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step2
Decide on the type of filter you want to apply based on your desired outcome. Some common filters include:

a. Averaging filter

b. Gaussian filter

c. Median filter

d. Laplacian filter

### Step3
A filter kernel is a small matrix that is applied to each pixel in the image to produce the filtered result. The size and values of the kernel determine the filter's behavior. For example, an averaging filter kernel has all elements equal to 1/N, where N is the kernel size.

### Step4
Use the library's functions to apply the filter to the image. The filtering process typically involves convolving the image with the filter kernel.

### Step5
Visualize the filtered image using a suitable method (e.g., OpenCV's imshow, Matplotlib). Save the filtered image to a file if needed.

## Program:
### Developed By    :   Sam Israel D
### Register Number :   212222230128
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
import cv2
import numpy as np

# Load the image
image = cv2.imread("nature.jpg")

# Create the averaging kernel
kernel = np.ones((3, 3)) / 9

# Apply the averaging filter
averaging_smoothed = cv2.filter2D(image, -1, kernel)

# Display the result
cv2.imshow("Averaging Smoothed", averaging_smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
ii) Using Weighted Averaging Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create the weighted averaging kernel
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16  # Normalized weights for better visualization

# Apply the weighted averaging filter
image3 = cv2.filter2D(image2, -1, kernel1)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3, cmap='gray')
plt.title("Weighted Average Filter Image")
plt.axis("off")

# Show the plot
plt.show()
```
iii) Using Gaussian Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with a kernel size of 5x5 and sigmaX (standard deviation) of 0 (calculated automatically)
gaussian_blur = cv2.GaussianBlur(image2, (5, 5), 0)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the Gaussian blurred image
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis("off")

# Show the plot
plt.show()
```
iv)Using Median Filter
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("nature.jpg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply median filter with a kernel size of 3x3
median = cv2.medianBlur(image2, 3)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the median filtered image
plt.subplot(1, 2, 2)
plt.imshow(median, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

# Show the plot
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("nature.jpg")

# Convert the image to RGB color space
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Create the Laplacian kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Apply the Laplacian kernel
image3 = cv2.filter2D(image2, -1, kernel)

# Create the figure and subplots
plt.figure(figsize=(10, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

# Display the Laplacian filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")

plt.show()
```
ii) Using Laplacian Operator
```Python
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
```

## OUTPUT:
### 1. Smoothing Filters

i) Using Averaging Filter
![image](./images/s1.png)

ii)Using Weighted Averaging Filter

![image](./images/s2.png)

iii)Using Gaussian Filter

![image](./images/s3.png)

iv) Using Median Filter

![image](./images/s4.png)

### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal

![image](./images/s5.png)

ii) Using Laplacian Operator

![image](./images/s6.png)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
