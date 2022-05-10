# Pneumonia Detection using Machine Learning
## Team Members
- Rugved Manookar To do
- Anjali Jain     015244151
- Adarsh Patil    To do

## Introduction
## Models
1. SVM
2. Random Forest
## Dataset and Preprocessing
Dataset : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

We start with 2000 samples from the Mendeley dataset.
1000 positive samples (x-rays labeled pneumonia)
1000 negative samples (x-rays labelled as not pneumonia)

These samples are high quality JPEG images. We use python OpenCV library to resize these images and then convert to grayscale inorder to reduce the complexity. All the pneumonia positive samples are labelled as 1 and negative samples are labelled as 0. 

-insert pneumonia positive sample
-insert pneumonia negative sample
-insert grayscale image for pneumoia postive sample


## Feature Extraction

## Feature Scaling
Feature Scaling is a technique of bringing down the values of all the independent features of our dataset on the same scale. We use StandardScaler from sklearn.preprocessing library to scale our feature vectors.

## Training and cross-validation
We are running both the model SVM and Random Forest for training the model on feature vector of training dataset. We are using K-Fold where K=5 cross validation to ensure there is no overfitting.

## Testing 
Post training the model, we fit our trained model on test dataset.Using sklearn.metrics we calculated precision, recall, f1 score and accuracy score.

## Comparision between SVM and Random Forest Model
1 - pneumonia
0 - Normal

###  Confusion Matrix for SVM Model
<img src="outputscreenshots/confusion_matrix_svm.png" width="500">

###  Confusion Matrix for Random Forest Model
<img src="outputscreenshots/confusion_matrix_random_forest.png" width="500">

## Integration with Flask App
## UI Screenshots
