import os
from collections import Counter
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image

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

def analyzeDataset(data_dir):
    findClassDistribution(data_dir)
    getImageStats(data_dir)
    #Add more exploration techniques

def findClassDistribution(data_dir):
    counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in os.listdir(data_dir)}
    print(f'\n----------\nClass distribution is the following:\n{counts}\n----------\n')

def getImageStats(data_dir, max_images=500):
    dims = []
    image_count = 0
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            path = os.path.join(cls_path, fname)
            try:
                with Image.open(path) as img:
                    dims.append(img.size)
                    image_count += 1
                    if image_count >= max_images:
                        break
            except:
                continue
        if image_count >= max_images:
            break
    dims = np.array(dims)
    avg_size = tuple(np.mean(dims, axis=0).astype(int))
    min_size = tuple(np.min(dims, axis=0))
    max_size = tuple(np.max(dims, axis=0))
    most_common = Counter(map(tuple, dims)).most_common(1)[0]

    print(f"Total images scanned: {len(dims)}")
    print(f"Average image size:   {avg_size}")
    print(f"Smallest image size: {min_size}")
    print(f"Largest image size:  {max_size}")
    print(f"Most common size:    {most_common[0]} (count: {most_common[1]})")

def loadAllDatasets(dataset_path, batch_size=None, display_analysis=True):
    training_path = os.path.join(dataset_path, "base_model_data/Train_Set_Folder")
    val_path = os.path.join(dataset_path, "base_model_data/Validation_Set_Folder")
    testing_path = os.path.join(dataset_path, "base_model_data/Test_Set_Folder")

    #Analyze datasets for image sizes and distribution
    if display_analysis:
        print(f'\n----------\nAnalyzing Training Data\n----------\n')
        analyzeDataset(training_path)
        print(f'\n----------\nAnalyzing Validation Data\n----------\n')
        analyzeDataset(val_path)
        print(f'\n----------\nAnalyzing Testing Data\n----------\n')
        analyzeDataset(testing_path)

    training_data = loadDataset(training_path, batch_size=batch_size, shuffle=True, image_size=(224,224))
    val_data = loadDataset(val_path, batch_size=batch_size, image_size=(224,224))
    testing_data = loadDataset(testing_path, batch_size=batch_size, image_size=(224,224))

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

def getEdgesFromGreenness(image, use_mpn=False):
    #Get a greenness mask from threshold, then apply to retrieve only leaf
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if use_mpn:
            #Midpoint normalization: [-1, 1]
            #Converts to [0, 1] by centering around [-.5, .5], then multiplies by 2
            mpn = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
            #Rescale to [0, 255] to keep same downstream flow
            image = ((mpn + 1.0) / 2.0) * 255.0
            #Needed to convert back to uint 8 for cv2 to work
            image = image.astype(np.uint8)

    mask = getGreennessMask(image)
    leaf_only = cv2.bitwise_and(image, image, mask=mask)

    #Convert to grayscale for CLAHE
    gray_leaf = cv2.cvtColor(leaf_only, cv2.COLOR_RGB2GRAY)
    if not use_mpn:
        #Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_leaf = clahe.apply(gray_leaf)
        return enhanced_leaf

    return gray_leaf

def is_green_enough(image_np, threshold=0.33):
    #separate RGB and determine if image is green enough to use greenness
    red = image_np[:, :, 0].astype(np.float32)
    green = image_np[:, :, 1].astype(np.float32)
    blue = image_np[:, :, 2].astype(np.float32)
    total = red + green + blue + 1e-6  # prevent division by zero
    green_ratio = np.mean(green / total)
    return green_ratio > threshold


def adaptive_segmentation(image_np, use_mpn=False):
    if is_green_enough(image_np):
        return getEdgesFromGreenness(image_np, use_mpn=use_mpn)
    else:
        return handMadeSegmentation(image_np, 110)

######################
# Creates data pipeline for returning zipped dataset of raw (images we got from Kaggle) and segmented data
def create_dual_input_dataset(dataset, segmentation_fn):
    def process(image, label):
        def segment(image_np):
            image_np = image_np.astype(np.uint8)

            # Ensure RGB shape by checking dimmensions and grayscale channel
            if image_np.ndim == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[-1] == 1:
                image_np = cv2.cvtColor(np.squeeze(image_np, axis=-1), cv2.COLOR_GRAY2RGB)

            #Apply segmentation function of choice
            mask = segmentation_fn(image_np)

            #Normalize mask to [0, 1]
            mask = mask.astype(np.float32) / 255.0
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.clip(np.nan_to_num(mask), 0.0, 1.0)

            #Preprocess image for VGG19 but have to convert to NP float 32 first
            image_np = image_np.astype(np.float32)
            image_np = preprocess_input(image_np)  # Note: doesn't normalize to [0, 1]

            return image_np, mask

        #Apply function using tensorflow.numpy_function
        raw, mask = tensorflow.numpy_function(
            func=segment,
            inp=[image],
            Tout=(tensorflow.float32, tensorflow.float32)
        )

        #Set shapes manually so that model TF can use
        raw.set_shape([224, 224, 3])
        mask.set_shape([224, 224, 1])
        label.set_shape([])

        #Must return in tuple of raw and mask for multiple inputs
        return (raw, mask), label

    #Apply mapping and batching AFTER segmentation - this allows us to process each image indivudually
    return dataset.map(process, num_parallel_calls=tensorflow.data.AUTOTUNE).batch(32).prefetch(tensorflow.data.AUTOTUNE)


######################
#Display a given image with the given title
def displayImage(image, title="Image"):
    plt.title(title)
    plt.imshow(image)
    plt.show()

######################
#Display all images that have been preprocessed
def displayProcessedImages(dataset, num_examples):
    for (inputs, labels) in dataset.take(1):
        raw_images = inputs[0].numpy()
        mask_images = inputs[1].numpy()
        labels = labels.numpy()
        break

    #Plot: 5 rows, 2 columns (Raw | Mask)
    plt.figure(figsize=(8, num_examples * 2))

    for i in range(num_examples):
        #Raw Image (Preprocessed with VGG19)
        plt.subplot(num_examples, 2, i * 2 + 1)
        plt.imshow(raw_images[i])
        plt.title(f"Raw Image- VGG19 Preprocessed\nLabel: {labels[i]}")
        plt.axis('off')

        #Mask
        plt.subplot(num_examples, 2, i * 2 + 2)
        plt.imshow(mask_images[i].squeeze(), cmap='gray')
        plt.title("Masked Image")
        plt.axis('off')

    plt.tight_layout()
    plt.show()