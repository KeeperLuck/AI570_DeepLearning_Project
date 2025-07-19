import matplotlib
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn
from sklearn.metrics import precision_recall_fscore_support

#############################
# "Random guessing" means picking the class at random
# We can estimate what the loss of a model that randomly guesses by 
# calculating log(number_of_possible_classes)
# So if we had 15 possible plant classifications,
# a model that guesses randomly would have a loss of about log(15)
def calculateRandomGuessThreshold(num_classes):
    return np.log(num_classes)

#############################
# Graph the given accuracy statistics (i.e. training accuracy)
# The "goal" is the threshold for success we set in the project proposal.
# I think it was 75% (0.75), which was pretty low
def graphAccuracy(accuracy, goal, title="Accuracy"):
    plt.plot(accuracy, label="Accuracy")
    plt.title(title)
    plt.axhline(y=goal, color="red", linestyle='--', label="Goal Accuracy") # Horizontal dotted line
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()

#############################
# Graph the loss (i.e. training loss)
# Also draw the "Random Guessing" y-line
def graphLoss(loss, num_classes, title="Loss"):
    random_guess_threshold= calculateRandomGuessThreshold(num_classes)
    plt.plot(loss, label="Loss")
    plt.axhline(y=random_guess_threshold, color="red", linestyle='--', label="Random Guess Threshold") # Horizontal dotted line
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Amount")
    plt.grid()
    plt.legend()
    plt.show()

#############################
# Graph the confusion matrix based on the testing predictions and the true labels
# True values will be the y-axis, predictions will be the x-axis
# A perfect model will have a perfectly dark blue diagonal line
def graphConfusionMatrix(labels, predictions, class_names):
    predictions= turnPredictionsIntoIndeces(predictions)
    conf_mat= sklearn.metrics.confusion_matrix(labels, predictions)
    plt.figure(figsize=(20,18)) # We have so many classes that if we don't make the figure bigger everything will overlap
    seaborn.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=60)
    plt.show()

#############################
# The predictions returned by testing the model is a list of confidences for each possible class
# The highest confidence is the the class the model thinks the image belongs to
# Figure out which class the prediction is saying by choosing the index holding the highest confidence
# EX: [0.02, 0.3, 0.4, 0.28] ---> 0.4 is the highest --> index 2
def turnPredictionsIntoIndeces(predictions):
    indeces= []
    for prediction in predictions:
        prediction= list(prediction)
        maximum= max(prediction)
        index= prediction.index(maximum)
        indeces.append(index)
    return indeces

#############################
# The predictions returned by the testing model is a list of confidences for each possible class
# Get the max of each tests, so we can just ignore the rest of the irrelevant values
# EX: [0.02, 0.3, 0.4, 0.28] ---> 0.4
def turnPredictionsIntoConfidences(predictions):
    confidences= []
    for prediction in predictions:
        confidence= max(prediction)
        confidences.append(confidence)
    return confidences
    
#############################
# Calculate the precision, recall, fscore, and support
# One value for each class our model can predict
# i.e. 30 classes ---> list of 30 values for precision, recall, etc.
def getTestData(predictions, labels):
    num_classes= len(predictions[0])
    predictions= turnPredictionsIntoIndeces(predictions)
    precision, recall, fscore, support= precision_recall_fscore_support(labels, predictions, labels=range(num_classes))
    return precision, recall, fscore, support
    
#############################
# Create a bar graph based on the given data
def createBarGraph(data, class_names, metric_name, title="Bar Graph"):
    x= np.arange(len(data))
    plt.figure(figsize=(15,6)) # If we don't make the graph bigger, everything will be too crowded
    plt.bar(x, data, color='green')
    plt.xlabel('Class')
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xticks(x, class_names, rotation=60)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


