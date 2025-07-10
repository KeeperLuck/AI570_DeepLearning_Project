import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

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
        def trainModel(self, features, epochs=5):
                if self.model is None:
                        print("Unable to train model, model is None!")
                        return None
                
                results= self.model.fit(features, epochs=epochs)
                return results
        
        ###############
        # Test the model
        def testModel(self, features):
                if self.model is None:
                        print("Unable to test model, model is None!")
                        return None
                
                predictions= self.model.predict(features)
                return predictions



if __name__ == "__main__":
        model= BaseModel()



