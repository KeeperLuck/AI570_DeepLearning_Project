import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow
import numpy as np

######################
# Load data in the simplest way possible
# Returns the dataset (if possible)
def loadDataset(dataset_path, image_size=(128,128), batch_size=32, shuffle=False):
    try:
        data= tensorflow.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return data
    except Exception as E:
        print(f"Unable to get dataset at location {dataset_path}:\n\n{E}")
        return None

######################
# Get a single image and return it as a CV2 image
def loadImage(image_path):
    try:
        image= cv2.imread(image_path)
        return image
    except Exception as E:
        print(f"Unable to load image at location [{image_path}]")
        return None

######################
# Peform threshold segmentation on the given CV2 image
def thresholdSegmentation(image, threshold=125):
    # Turn the image grey
    greyscale_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Segment the image
    ignore, segmented_image= cv2.threshold(greyscale_image, threshold, 255, cv2.THRESH_BINARY)
    return segmented_image

######################
# Perform edge segmentation on the given CV2 image
def edgeSegmentation(image, kernel_size=(5,5), threshold=100):
    # Make the image grey
    greyscale_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply a Gaussian blur to the photo
    blurred_image= cv2.GaussianBlur(greyscale_image, kernel_size, 0)
    # Segment the photo
    segmented_image= cv2.Canny(blurred_image, threshold1=threshold, threshold2=threshold*2)
    return segmented_image

######################
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

######################
# This is a segmentation algorithm Lin Et Al. Applied for greenness thresholds
# from one of the papers for my Literature Review and will return a greenness mask to be applied
def getGreennessMask(image, greenness_thresh=0.1):
    #Convert to HSV like lin et al. to create binary mask
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Normalize H, S, V to float values in range of [0,1]
    h = h.astype(np.float32)
    s = s.astype(np.float32) / 255
    v = v.astype(np.float32) / 255

    #Calculate greenness score for each pixel.
    #Hue for green is around 60 in HSV. So we will penalize distance from 60
    #s = saturation, v = value or brightness, and h = hue, so we will find the distance from
    #green and then normalize to be between 0 and 1.
    greenness = s * v * (1 - np.abs(h - 60) / 60)
    greenness = np.clip(greenness, 0, 1)

    #Create and return binary mask from greenness score
    mask = (greenness >= greenness_thresh).astype(np.uint8) * 255
    return mask

def getEdgesFromGreenness(image):
    #Get a greenness mask from threshold, then apply to retrieve only leaf
    mask = getGreennessMask(image)
    leaf_only = cv2.bitwise_and(image, image, mask=mask)

    #Convert to grayscale for CLAHE
    gray_leaf = cv2.cvtColor(leaf_only, cv2.COLOR_RGB2GRAY)

    #Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_leaf = clahe.apply(gray_leaf)

    return enhanced_leaf

######################
# Display a given image with the given title
def displayImage(image, title="Image"):
    plt.title(title)
    plt.imshow(image)
    plt.show()