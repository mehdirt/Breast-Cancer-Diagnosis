import pandas as pd
from sklearn.preprocessing import StandardScaler

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