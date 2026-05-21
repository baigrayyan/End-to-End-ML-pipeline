# End-to-End-ML-pipeline
## Objective
To build a reusable and production-ready machine learning pipeline for predicting customer churn.

## Dataset
Telco Churn Dataset (simulated or loaded from a URL for demonstration purposes).

## Pipeline Steps
The ML pipeline follows these key steps:

1.  **Data Loading**: The Telco Churn dataset is loaded into a pandas DataFrame.
2.  **Data Preprocessing**:
    *   `TotalCharges` column is converted to a numeric type, with missing values handled by median imputation.
    *   The target variable `Churn` is encoded from 'Yes'/'No' to 1/0.
    *   `customerID` is dropped as it's an identifier.
    *   The dataset is split into training and testing sets (80/20 split).
3.  **Feature Identification**: Numerical and categorical features are identified for differential preprocessing.
4.  **Preprocessing Pipelines**: Scikit-learn `ColumnTransformer` is used to create a preprocessor:
    *   `StandardScaler` is applied to numerical features.
    *   `OneHotEncoder` is applied to categorical features.
5.  **ML Pipelines**: Complete pipelines are built combining the preprocessor with two different classification models:
    *   **Logistic Regression**
    *   **Random Forest Classifier**
6.  **Hyperparameter Tuning**: `GridSearchCV` is employed to find the best hyperparameters for both Logistic Regression and Random Forest models.
7.  **Model Evaluation**: The best-performing models from GridSearchCV are evaluated on the unseen test set using metrics like accuracy, classification report, and confusion matrix.
8.  **Model Export**: The best overall pipeline (based on cross-validation performance) is exported using `joblib` for future use.

## How to Use the Exported Pipeline

To use the saved model for making predictions on new data, follow these steps:

1.  **Load the Pipeline**: Load the `.joblib` file using `joblib.load()`.
2.  **Prepare New Data**: Ensure your new data has the same column structure and data types as the training data, *excluding* the target variable.
3.  **Make Predictions**: Use the loaded pipeline's `predict()` method to get predictions.

### Example Code to Load and Predict:

```python
import joblib
import pandas as pd

# Load the saved pipeline (replace 'random_forest_pipeline.joblib' if Logistic Regression was chosen)
loaded_pipeline = joblib.load('random_forest_pipeline.joblib')

# Example of new data (replace with your actual new data)
# Make sure the columns match the original training data (excluding 'Churn')
new_data = pd.DataFrame({
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [5],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['DSL'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['Yes'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [70.0],
    'TotalCharges': [350.0]
})

# Make predictions
predictions = loaded_pipeline.predict(new_data)

print("Predictions on new data:", predictions)
# Output will be an array of 0s (no churn) or 1s (churn)
