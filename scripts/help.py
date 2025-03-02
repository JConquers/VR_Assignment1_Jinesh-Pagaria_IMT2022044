import cv2
import matplotlib.pyplot as plt

# This code reads an image and generates a histogram showing distribution of intensities


image = cv2.imread('../input/img2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (17, 17), 8)


if image is None:
    print("Error: Image not found or could not be loaded.")
    exit()

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure()
plt.title('Grayscale Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()
