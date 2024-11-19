### Welcome,



In this project, I’m building a machine learning model to predict whether a person has Autism Spectrum Disorder (ASD) based on several features like age, gender, and other attributes. The process involves:
1. **Exploring and Cleaning the Data**: Understanding and preparing the data for training.
2. **Training the Model**: Trying different algorithms to find the best one.
3. **Fine-Tuning**: Optimizing the model’s performance.
4. **Evaluating and Saving**: Testing the model's accuracy and saving it for future use.

In the end, I will give you the option to input your own values, and the model will predict whether the input indicates ASD or not. Let’s dive in.

---

### **1. Importing Libraries**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
```

**I begin by importing necessary libraries.**  
Think of this as getting all the tools I need for the job. I need libraries for handling data (`pandas`, `numpy`), building machine learning models (`DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`), and evaluating model performance. I also use **SMOTE** to handle class imbalance and **pickle** to save the trained model so you can use it later.

---

### **2. Exploring the Data**

```python
# Outliers, Class Imbalance, Preprocessing
# Identifying outliers, handling class imbalance, and preprocessing categorical features
```

**Before training the model, I take a good look at the data.**  
I make sure there are no outliers or class imbalances that could make the model biased. I also convert any categorical variables into a format that the model can understand (numerical).

---

### **3. Dealing with Outliers**

```python
def replace_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[column].median()
    df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df
```

**I deal with outliers that might distort the model.**  
Outliers are extreme values in the data (like an unusually high age) that can affect how the model performs. I replace them with the median of that feature, which is a more reasonable representation of the data.

---

### **4. Splitting the Data**

```python
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Next, I split the data into training and testing sets.**  
This is like training the model with some data and testing it on unseen data to evaluate how well it performs. I use 80% for training and 20% for testing.

---

### **5. Handling Class Imbalance with SMOTE**

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

**I balance the classes using SMOTE.**  
Sometimes, there are more examples of one class than another, which can make the model biased. SMOTE creates synthetic data for the underrepresented class, ensuring the model learns from both classes equally.

---

### **6. Trying Different Models**

```python
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} Cross-Validation Accuracy: {np.mean(scores):.2f}")
```

**I try different models to see which one works best.**  
I use three models: Decision Trees, Random Forests, and XGBoost. For each one, I evaluate its accuracy through cross-validation, which ensures I get reliable performance results.

---

### **7. Fine-Tuning the Best Model**

```python
param_grid_dt = {...}
param_grid_rf = {...}
param_grid_xgb = {...}

random_search_dt = RandomizedSearchCV(estimator=decision_tree, param_distributions=param_grid_dt, n_iter=20, cv=5, scoring="accuracy", random_state=42)
random_search_rf = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid_rf, n_iter=20, cv=5, scoring="accuracy", random_state=42)
random_search_xgb = RandomizedSearchCV(estimator=xgboost_classifier, param_distributions=param_grid_xgb, n_iter=20, cv=5, scoring="accuracy", random_state=42)
```

**Once I know which model is best, I fine-tune it for better performance.**  
I adjust the hyperparameters of the chosen model using **RandomizedSearchCV**, which helps me find the optimal settings to improve its accuracy.

---

### **8. Choosing the Best Model**

```python
best_model = None
best_score = 0

if random_search_dt.best_score_ > best_score:
    best_model = random_search_dt.best_estimator_
    best_score = random_search_dt.best_score_

if random_search_rf.best_score_ > best_score:
    best_model = random_search_rf.best_estimator_
    best_score = random_search_rf.best_score_

if random_search_xgb.best_score_ > best_score:
    best_model = random_search_xgb.best_estimator_
    best_score = random_search_xgb.best_score_

print(f"Best Model: {best_model} with Accuracy: {best_score:.2f}")
```

**I pick the best model after fine-tuning.**  
I compare the performance of all the models and choose the one that performs the best.

---

### **9. Evaluating the Model**

```python
y_pred = best_model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

**Now, I evaluate the model’s performance on new data.**  
I test the best model using the test set and evaluate how well it classifies new data using accuracy, precision, recall, and F1-score.

---

### **10. Saving the Model**

```python
with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)
```

**Finally, I save the trained model.**  
Using **pickle**, I save the model so that it can be reused later without retraining it.

---

### **11. Allowing User Input for Prediction**

Now comes the fun part – **user interaction**! After training the model, I give the user an option to input their own data and make predictions. Here's how I set it up:

```python
def predict_asd(input_data):
    # Load the trained model
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    
    # Process input_data in the same way as the training data
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Map prediction to readable result
    if prediction == 1:
        return "Predicted: ASD (Autism Spectrum Disorder)"
    else:
        return "Predicted: No ASD"

# Example usage:
input_data = {
    'Age': 25, 
    'Gender': 1, 
    'Result1': 100, 
    'Result2': 90, 
    'Result3': 85
}

print(predict_asd(input_data))
```

**I now allow the user to input their own values to make predictions.**  
The function `predict_asd` accepts user input (like age, gender, and test results), processes it in the same way as the training data, and uses the saved model to predict if the input data indicates ASD or not.

### **Try It Yourself!**
To make your own prediction, input the values (like Age, Gender, and test results) and the model will return whether the person is predicted to have ASD or not.

---

### END
