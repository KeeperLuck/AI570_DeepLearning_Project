import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy

def loadImage(image_path):
    try:
        image= cv2.imread(image_path)
        return image
    except Exception as E:
        print(f"Unable to load image at location [{image_path}]")
        return None

def thresholdSegmentation(image, threshold=125):
    greyscale_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ignore, segmented_image= cv2.threshold(greyscale_image, threshold, 255, cv2.THRESH_BINARY)
    return segmented_image

def edgeSegmentation(image, kernel_size=(5,5), threshold=100):
    greyscale_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image= cv2.GaussianBlur(greyscale_image, kernel_size, 0)
    segmented_image= cv2.Canny(blurred_image, threshold1=threshold, threshold2=threshold*2)
    return segmented_image

# This is a segmentation algorithm I read about in one of the 
# papers for my Literature Review
def handMadeSegmentation(image, threshold=150):
    # convert image to greyscale
    greyscale_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # convert greyscale to binary image
    binary_image= cv2.adaptiveThreshold(greyscale_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 10
    )

    # Apply a laplacian filter
    laplacian_image= cv2.Laplacian(binary_image, cv2.CV_64F) 

    # Invert the image
    inverted_image= cv2.bitwise_not(laplacian_image)
    return inverted_image


def displayImage(image, title="Image"):
    plt.title(title)
    plt.imshow(image)
    plt.show()