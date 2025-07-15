import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# Basic model, should be very minimal and Sequential
# Just a starting point
class BaseModel:
        def __init__(self):
                self.model= None
                return

        ###############
        # Create each layer and build the model as a whole
        def buildModel(self, layers:list):
                if layers is None or layers == []:
                        print(f"You gave bad layers: {layers}, unable to build model")
                        return
                
                self.model= keras.models.Sequential(layers)
                print("Model built")
                return
        
        ###############
        # Compile the model
        def compileModel(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]):
                if self.model is None:
                        print("Unable to build model, model is None!")
                        return
                
                self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                print("Model compiled")
                return
        
        ###############
        # Train the model 
        def trainModel(self, features, epochs=5, val_data = None):
                if self.model is None:
                        print("Unable to train model, model is None!")
                        return None
                if val_data is None:
                        results= self.model.fit(features, epochs=epochs)
                else:
                        results = self.model.fit(features, epochs=epochs, validation_data=val_data)
                return results
        
        ###############
        # Test the model
        def testModel(self, features):
                if self.model is None:
                        print("Unable to test model, model is None!")
                        return None

                predictions = self.model.predict(features)

                # Get true labels
                true_labels = []
                for _, label in features:
                        true_labels.extend(label.numpy())
                true_labels = np.array(true_labels)

                # Convert softmax to class index
                predicted_classes = np.argmax(predictions, axis=1)
                if true_labels.ndim > 1:
                        true_labels = np.argmax(true_labels, axis=1)
                acc = accuracy_score(true_labels, predicted_classes)
                print("Test Accuracy:", acc)
                print(classification_report(true_labels, predicted_classes))
                return predictions


if __name__ == "__main__":
        model= BaseModel()



