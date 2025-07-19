import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow
from tensorflow.keras.applications.vgg19 import preprocess_input
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
            shuffle=shuffle,
            label_mode='int'
        )
        return data
    except Exception as E:
        print(f"Unable to get dataset at location {dataset_path}:\n\n{E}")
        return None

def loadAllDatasets(dataset_path, batch_size=None):
    training_path = os.path.join(dataset_path, "base_model_data/Train_Set_Folder")
    val_path = os.path.join(dataset_path, "base_model_data/Validation_Set_Folder")
    testing_path = os.path.join(dataset_path, "base_model_data/Test_Set_Folder")

    training_data = loadDataset(training_path, batch_size=batch_size, shuffle=True)
    val_data = loadDataset(val_path, batch_size=batch_size)
    testing_data = loadDataset(testing_path, batch_size=batch_size)

    class_names = training_data.class_names

    return training_data, val_data, testing_data, class_names
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
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
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
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    mask = getGreennessMask(image)
    leaf_only = cv2.bitwise_and(image, image, mask=mask)

    #Convert to grayscale for CLAHE
    gray_leaf = cv2.cvtColor(leaf_only, cv2.COLOR_RGB2GRAY)

    #Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_leaf = clahe.apply(gray_leaf)

    return enhanced_leaf

######################
# Creates data pipeline for returning zipped dataset of raw (images we got from Kaggle) and segmented data
def create_dual_input_dataset(dataset, segmentation_fn):
    def process(image, label):
        def segment(image_np):
            image_np = image_np.astype(np.uint8)

            # Ensure RGB shape
            if image_np.ndim == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[-1] == 1:
                image_np = cv2.cvtColor(np.squeeze(image_np, axis=-1), cv2.COLOR_GRAY2RGB)

            # Apply segmentation function of choice
            mask = segmentation_fn(image_np)

            # Normalize mask to [0, 1]
            mask = mask.astype(np.float32) / 255.0
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.clip(np.nan_to_num(mask), 0.0, 1.0)

            # Preprocess image for VGG19 but have to convert to NP float 32 first
            image_np = image_np.astype(np.float32)
            image_np = preprocess_input(image_np)  # Note: doesn't normalize to [0, 1]

            return image_np, mask

        # Apply function using tensorflow.numpy_function
        raw, mask = tensorflow.numpy_function(
            func=segment,
            inp=[image],
            Tout=(tensorflow.float32, tensorflow.float32)
        )

        # Set shapes manually so that model TF can use
        raw.set_shape([128, 128, 3])
        mask.set_shape([128, 128, 1])
        label.set_shape([])

        #Must return in tuple of raw and mask for multiple inputs
        return (raw, mask), label

    # Apply mapping and batching AFTER segmentation - this allows us to process each image indivudually
    return dataset.map(process, num_parallel_calls=tensorflow.data.AUTOTUNE).batch(32).prefetch(tensorflow.data.AUTOTUNE)


######################
# Display a given image with the given title
def displayImage(image, title="Image"):
    plt.title(title)
    plt.imshow(image)
    plt.show()

######################
# Display all images that have been preprocessed
def displayProcessedImages(dataset, num_examples):
    for (inputs, labels) in dataset.take(1):
        raw_images = inputs[0].numpy()
        mask_images = inputs[1].numpy()
        labels = labels.numpy()
        break

    # Plot: 5 rows, 2 columns (Raw | Mask)
    plt.figure(figsize=(8, num_examples * 2))

    for i in range(num_examples):
        # Raw Image
        plt.subplot(num_examples, 2, i * 2 + 1)
        plt.imshow(raw_images[i])
        plt.title(f"Raw Image\nLabel: {labels[i]}")
        plt.axis('off')

        # Mask
        plt.subplot(num_examples, 2, i * 2 + 2)
        plt.imshow(mask_images[i].squeeze(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.show()