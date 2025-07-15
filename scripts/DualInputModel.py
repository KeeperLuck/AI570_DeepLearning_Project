from tensorflow.keras import Input, Model, layers
from BaseModel import BaseModel
from tensorflow.keras.applications import VGG19

class DualInputModel(BaseModel):
    def __init__(self, image_shape=(128, 128, 3), mask_shape=(128, 128, 1), num_classes=6):
        super().__init__()
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.num_classes = num_classes

    def buildModel(self):
        raw_input = Input(shape=self.image_shape, name='raw_input')
        #Adding transfer learner
        base_model = VGG19(include_top=False, weights='imagenet', input_tensor=raw_input)
        base_model.trainable = False
        #Set top layers to not trainable
        for layer in base_model.layers:
            layer.trainable = False

        x1 = base_model.output
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(128, activation='relu')(x1)

        #Segmented mask input path
        mask_input = Input(shape=self.mask_shape, name='mask_input')

        #3 x 3 kernel used in Conv2D can adjust later
        x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(mask_input)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D()(x2)

        x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D()(x2)

        x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.GlobalAveragePooling2D()(x2)

        x2 = layers.Dense(128, activation='relu')(x2)

        #Merge Layers and Final Classification
        merged = layers.Concatenate()([x1, x2])
        merged = layers.Dense(256, activation='relu')(merged)
        merged = layers.Dropout(0.3)(merged)
        merged = layers.Dense(128, activation='relu')(merged)
        merged = layers.Dropout(0.2)(merged)
        #Using softmax function for multiclass classification
        output = layers.Dense(self.num_classes, activation='softmax')(merged)

        self.model = Model(inputs=[raw_input, mask_input], outputs=output)
