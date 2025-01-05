from .data_utils import normalize_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def create_model(data):
    """"""
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

def test_model(model, X_test, y_test):
    """"""
    y_pred = model.predict(X_test)
    print(f"Accuracy of your model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification report: {classification_report(y_test, y_pred)}")
