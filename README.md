# Human Activity Recognizer

This repository contains a project to recognize human activities using machine learning techniques. The model is trained and tested on the **[UCI Human Activity Recognition dataset](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)**. The pipeline involves feature selection, classifier training, and testing.

## Files and Directories

- `HAR.ipynb`: Notebook containing the training code, including data preprocessing, feature selection, and model training.
- `test.ipynb`: Notebook used to test the trained model on new or unseen data.
- `utils.py`: Python script that contains utility functions such as `removeDuplicateColumns`, which is used to clean the dataset by removing redundant features.
- `model.pkl`: Trained Random Forest model saved in pickle format using `joblib`, which can be loaded and used in the `test.ipynb` for making predictions.

## Dataset

The dataset used in this project is the **UCI Human Activity Recognition dataset**, which can be downloaded from [this link](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones).

## Approach

The model training approach is organized as follows:

### 1. Label Encoding
- The activity labels (output) are encoded into numerical values to be used by the classifier.

### 2. Removing Duplicate Columns
- The dataset is checked for redundant columns, which are removed using the `removeDuplicateColumns` function from the `utils.py` file.

### 3. Feature Selection using `SelectKBest`
- `SelectKBest` is applied to select the top k most important features based on statistical tests.

### 4. Recursive Feature Elimination (RFE) with Random Forest Classifier
- Recursive Feature Elimination (RFE) is applied using the Random Forest Classifier to further refine the feature selection and remove the least important features iteratively.

### 5. Random Forest Classifier
- A Random Forest Classifier is trained using the selected features. Random Forest is chosen due to its robustness and ability to handle complex datasets effectively.

### Pipeline
- Steps 2 to 5 are organized into a pipeline, ensuring a streamlined and reproducible process for data preparation, feature selection, and model training.
  
### Exporting the Model
- The trained model is exported into a `.pkl` file using `joblib` for later use in the `test.ipynb` notebook for testing and predictions.

## How to Run

### Training the Model
1. Open and run `HAR.ipynb` to:
   - Preprocess the data.
   - Train the Random Forest model.
   - Export the trained model to a `.pkl` file.

### Testing the Model
1. Open and run `test.ipynb` to:
   - Load the saved model (`pipeline.pkl`).
   - Test the model on new data or the test set.
   
## Requirements

Make sure to install the following dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn joblib
