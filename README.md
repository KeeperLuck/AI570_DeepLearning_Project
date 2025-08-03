# Plant Species Classification with Convolutional Neural Networks
## AI-570 DeepLearning Project

## Team Members
- **Cody Gould**
- **Christopher Gilleo**

## Abstract
This project focuses on optimizing classification accuracy for 30 plant species by leveraging deep learning techniques. A dual-input convolutional neural network was built by combining a VGG19-based feature extractor for raw plant images (VGG19 preprocessed) with a custom CNN for segmented inputs. Segmentation was performed using greenness-based thresholds, adaptive segmentation, and Midpoint Normalization (Taufik et al., 2025). VGG19-preprocessed and segmented images were zipped into a dual-input tensor. Hyperparameter tuning was performed using Keras Tuner. Top-k accuracy, confusion matrices, F1 scores, recall, and precision were shown to evaluate performance. Training metrics were plotted against a random guess loss threshold and a 75% accuracy goal. The adaptive segmentation approach (DIH-A) with midpoint normalization yielded the best results, achieving a maximum training accuracy of 0.9749 and minimum loss of 0.0873 (13 epochs). Test accuracy peaked at 0.9533(15 epochs). Cassava and pineapple were classified with near-perfect accuracy, while cantaloupe and melon performed the worst due to visual similarity.

## Results
#### Training Accuracy
| **Approach**     | **Train Accuracy (Max)**      | **Train Loss (Min)**       |
|------------------|-------------------------------|-----------------------------|
| SDI              | 0.9600 (at epoch 10/10)       | 0.1195 (at epoch 10/10)     |
| DIH              | 0.9633 (at epoch 10/10)       | 0.1160 (at epoch 10/10)     |
| DIH-A (10ep)     | 0.9665 (at epoch 10/10)       | 0.1094 (at epoch 10/10)     |
| DIH-A (15ep)     | 0.9749 (at epoch 3/5)         | 0.0873 (at epoch 3/5)       |
| DIH-AM           | 0.9603 (at epoch 10/10)       | 0.1263 (at epoch 10/10)     |

#### Testing Accuracy
| **Approach**     | **Testing Accuracy** | **Minimum Class Accuracy** | **Maximum Class Accuracy** |
|------------------|----------------------|------------------------------------------------------|----------------------------|
| SDI              | 0.9383               | 76% (Cantaloupe)                                     | 100% (Pineapple, Watermelon) |
| DIH              | 0.9403               | 79% (Cantaloupe)                                     | 100% (Pineapple)           |
| DIH-A (10ep)     | 0.9440               | 77% (Cantaloupe)                                     | 100% (Pineapple)           |
| DIH-A (15ep)     | 0.9553               | 70% (Cantaloupe)                                     | 100% (Cassava)             |
| DIH-AM           | 0.9453               | 71% (Cantaloupe)                                     | 99% (Papaya, Pineapple, Watermelon) |

## How to Run
The code is written in Python using TensorFlow and Keras. Once installing dependencies, this project can be run as a Jupyter notebook. Simply step through each code block and analyze the results. Models have been saved in the repository, so they can be easily loaded utilizing Keras *(This may be beneficial if you do not have access to a GPU)*. Python files were created for utility functions and model architecture to ensure ease of reuse. 

### Dependencies
- TensorFlow (>=2.14)
- OpenCV
- NumPy
- Keras Tuner 
- matplotlib 
- seaborn
- keras_tuner 
- importlib 
- sklearn 
- cv2 

#### Install dependencies & configure environment:

pip install -r requirements.txt

**Note:** it may be best to run this project in a virtual environment with Tensorflow and cuda support. For Windows machines, WSL is recommended to ensure GPU support. 