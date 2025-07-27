from tensorflow.keras import Input, Model, layers, optimizers
from keras_tuner.tuners import RandomSearch
import tensorflow.keras as keras
from BaseModel import BaseModel
from tensorflow.keras.applications import VGG19

class DualInputModel(BaseModel):
    def __init__(self, image_shape=(224, 224, 3), mask_shape=(224, 224, 1), num_classes=6):
        super().__init__()
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        self.num_classes = num_classes
        self.tuner = None

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

    def buildHPModel(self, hp):
        raw_input = Input(shape=self.image_shape, name='raw_input')
        mask_input = Input(shape=self.mask_shape, name='mask_input')

        base_model = VGG19(include_top=False, weights='imagenet', input_tensor=raw_input)
        base_model.trainable = False

        #Set top layers to not trainable
        for layer in base_model.layers:
            layer.trainable = False

        x1 = base_model.output
        x1 = layers.GlobalAveragePooling2D()(x1)
        x1 = layers.Dense(hp.Int('dense1_units', 64, 256, step=64), activation='relu')(x1)

        x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(mask_input)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D()(x2)
        x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.MaxPooling2D()(x2)
        x2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.GlobalAveragePooling2D()(x2)
        x2 = layers.Dense(hp.Int('dense2_units', 64, 256, step=64), activation='relu')(x2)

        merged = layers.Concatenate()([x1, x2])
        merged = layers.Dense(hp.Int('dense3_units', 128, 512, step=128), activation='relu')(merged)
        merged = layers.Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1))(merged)
        merged = layers.Dense(hp.Int('dense4_units', 64, 256, step=64), activation='relu')(merged)
        merged = layers.Dropout(hp.Float('dropout2', 0.1, 0.4, step=0.1))(merged)

        output = layers.Dense(self.num_classes, activation='softmax')(merged)

        model = Model(inputs=[raw_input, mask_input], outputs=output)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_accuracy'),
            ]
        )
        self.model = model
        return model

    ###############
    # Running Keras Tuner to optimize the model of our choice, then returning best model over 10 trials
    def optimize_hyperparameters(self, train, val, test, model_builder=None):
        def build_tuner_model(hp):
            self.model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate', [.01, .001, .0001])
                ),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        if model_builder is None:
            model_builder = build_tuner_model

        # Search for best out of 10 trials and save results in "plant_tuner" directory for future use
        tuner = RandomSearch(
            model_builder,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=1,
            directory='plant_tuner',
            project_name='plant_model'
        )

        tuner.search(train, validation_data=val, epochs=10)
        self.tuner = tuner
        tuner.results_summary()
        best_model = tuner.get_best_models(num_models=1)[0]
        self.model = best_model
        return best_model
