from .data_utils import normalize_data

import numpy as np # Imported for annotations
import pandas as pd # Imported for annotations
from sklearn.base import BaseEstimator # Imported for annotations
from sklearn.preprocessing import StandardScaler # Imported for annotations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def create_model(data: pd.DataFrame) -> tuple[BaseEstimator, StandardScaler]:
    """
    Create, train, and test a logistic regression model.

    Steps:
    1. Separates features (X) and labels (y) from the dataset.
    2. Normalizes the features using a scaler.
    3. Splits the data into training and testing sets.
    4. Trains a Logistic Regression model on the training data.
    5. Tests the trained model on the test data.

    Args:
        data (pd.DataFrame): The cleaned dataset.

    Returns:
        tuple: A tuple containing:
            - BaseEstimator: The trained Logistic Regression model.
            - StandardScaler: The scaler object used for normalization.
    """
    # Seperate features and labels 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Preprocess the data
    X, scalar = normalize_data(X) # Normalizing data

    # Split the data to train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) 

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    test_model(model, X_test, y_test)

    return model, scalar

def test_model(
    model: BaseEstimator, 
    X_test: np.ndarray, 
    y_test: pd.Series
) -> None:
    """
    Test a trained model and print its performance metrics.

    Args:
        model (BaseEstimator): The trained model to evaluate.
        X_test (np.ndarray): Test feature set.
        y_test (pd.Series): True labels for the test set.

    Prints:
        - Accuracy score of the model.
        - Classification report with precision, recall, F1-score, and support.
    """
    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Get detailed classification report
    report = classification_report(y_test, y_pred)

    print(f"Accuracy of your model: {accuracy}")
    print(f"Classification report: {report}")
