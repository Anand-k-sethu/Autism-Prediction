# Autism Detection using Machine Learning

## Project Overview
This project aims to build a predictive model for detecting autism based on features derived from a dataset. The goal is to train, validate, tune, and evaluate three tree-based machine learning models (Decision Tree, Random Forest, and XGBoost) to determine if an individual shows signs of autism. The process involves hyperparameter tuning through RandomizedSearchCV and evaluating the best model on both the training and test datasets.

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Development and Tuning](#model-development-and-tuning)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

## Project Structure
```
autism_detection/
│
├── LICENSE                    # Contains the legal terms for using and distributing the project.
├── README.md                   # Provides an overview of the project, installation, usage, and contribution guidelines.
│
└── predicting-autism-using-ml.ipynb  # Jupyter notebook implementing the machine learning workflow for autism detection.

```

## Prerequisites
Ensure you have the following installed before proceeding:
- Python (>=3.7)
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Pickle

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/autism_detection.git
   cd autism_detection
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the necessary packages:
   ```sh
   pip install pandas numpy scikit-learn xgboost
   ```

## Data Preparation
### Dataset
The project uses a dataset containing features associated with autism. Below are some key columns:

- **A1_Score** to **A10_Score**: Numerical scores associated with different aspects of autism detection.
- **age**: The age of the subject.
- **gender**: Categorical (Male, Female).
- **ethnicity**: Categorical (e.g., Asian, Black).
- **jaundice**: Categorical (Yes, No).
- **austism**: Categorical (Yes, No).
- **Country of residence**: Categorical (e.g., USA, India).
- **used_app_before**: Categorical (Yes, No).
- **relation**: Categorical (Self, Parent).

### Preprocessing
- The data is split into training and testing sets.
- SMOTE (Synthetic Minority Over-sampling Technique) is applied on the training set to balance the class distribution.

## Model Development and Tuning
### Part 1: Cross-Validation
The project uses 5-fold cross-validation to evaluate the Decision Tree, Random Forest, and XGBoost models. Below are the accuracies obtained through cross-validation:

- **Decision Tree**: Accuracy scores range from 0.79 to 0.88 (average ~0.84).
- **Random Forest**: Accuracy scores range from 0.91 to 0.93 (average ~0.92).
- **XGBoost**: Accuracy scores range from 0.87 to 0.92 (average ~0.91).
- **Conclusion**: Random Forest outperformed the other models with the highest cross-validation accuracy.

### Part 2: Hyperparameter Tuning
Hyperparameter tuning is performed using `RandomizedSearchCV` for each of the three models:
- **Decision Tree**: Explores `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
- **Random Forest**: Explores `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`.
- **XGBoost**: Explores `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

The `RandomizedSearchCV` returns the best set of hyperparameters for each model, optimizing their performance.

### Part 3: Best Model Selection
The best-performing model across Decision Tree, Random Forest, and XGBoost based on cross-validation scores is selected and saved to `best_model.pkl`.

## Model Evaluation
The final selected model (Random Forest) is evaluated on the test set. Below are the results:
- **Accuracy score:** ~81.88%
- **Confusion Matrix:** Shows the number of true positive, false positive, true negative, and false negative predictions.
- **Classification Report:** Provides detailed metrics (precision, recall, F1-score) for both classes.

## Usage
To make predictions on new data:
1. Load the saved model and encoders using:
   ```python
   import pickle
   import pandas as pd

   def load_resources():
       with open("best_model.pkl", "rb") as f:
           model = pickle.load(f)
       with open("encoders.pkl", "rb") as f:
           encoders = pickle.load(f)
       return model, encoders

   def encode_input_data(input_data, encoders):
       input_df = pd.DataFrame([input_data])
       for feature in ['gender', 'ethnicity', 'jaundice', 'austism', 'Country of residence', 'used_app_before', 'relation']:
           input_df[feature] = encoders[feature].transform(input_df[feature])
       return input_df

   def predict(input_data):
       model, encoders = load_resources()
       input_data = encode_input_data(input_data, encoders)
       prediction = model.predict(input_data)
       return prediction[0]

   input_data = {
       'A1_Score': 5,
       'A2_Score': 6,
       'A3_Score': 7,
       'A4_Score': 4,
       'A5_Score': 3,
       'A6_Score': 2,
       'A7_Score': 5,
       'A8_Score': 6,
       'A9_Score': 7,
       'A10_Score': 8,
       'age': 10,
       'gender': 'm',
       'ethnicity': 'Asian',
       'jaundice': 'yes',
       'austism': 'no',
       'Country of residence': 'Canada',
       'used_app_before': 'yes',
       'relation': 'Self'
   }

   prediction = predict(input_data)

   result_text = "No Autism Detected" if prediction == 0 else "Autism Detected"
   print(f"Prediction: {result_text}")
   ```

## Results and Analysis
The final Random Forest model achieved an accuracy of ~81.88% on the test dataset. The confusion matrix and classification report further highlight the model’s performance across both classes. Despite the good overall accuracy, there is potential for improvement, especially in the recall and F1-score for class 1 (autism detected).

## Future Improvements
- **Class Imbalance Handling**: Consider applying techniques like SMOTE or cost-sensitive learning to improve performance for class 1.
- **Hyperparameter Optimization**: Further tune hyperparameters or experiment with different model architectures (e.g., ensemble methods or deep learning techniques).
- **Feature Engineering**: Introduce new features or transformations to enhance model performance.
- **Stratified Cross-Validation**: Ensure better representation of class distribution in the cross-validation process.

---


---
Thanks for reading
