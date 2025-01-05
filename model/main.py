import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def get_data(path):
    """"""
    data = pd.read_csv(path)
    return data

def clean_data(data):
    """"""
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def normalize_data(data):
    """"""
    # Normalize the data
    scalar = StandardScaler()
    data = scalar.fit_transform(data)

    return data, scalar

def test_model(model, X_test, y_test):
    """"""
    y_pred = model.predict(X_test)
    print(f"Accuracy of your model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification report: {classification_report(y_test, y_pred)}")


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

def main() -> None:
    # Get the clean data
    dframe = get_data(path="data/data.csv")
    dframe = clean_data(dframe)

    # Fit the model
    model, scalar = create_model(dframe)

    # Save and export the model
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)



if __name__ == '__main__':
    main()