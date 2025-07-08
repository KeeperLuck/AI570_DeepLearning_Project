import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# Basic model, should be very minimal
# Just a starting point
class BaseModel:
        def __init__(self):
                self.model= None
                return

        ###############
        # Create each layer and build the model as a whole
        def buildModel(self):
                return
        
        ###############
        # Compile the model
        def compileModel(self):
                return
        
        ###############
        # Train the model 
        def trainModel(self):
                return
        
        ###############
        # Test the model
        def testModel(self):
                return



if __name__ == "__main__":
        model= BaseModel()



