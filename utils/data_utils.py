import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        path (str): The file path to the CSV dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    data = pd.read_csv(path)
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.

    Steps:
    - Drops unnecessary columns ('Unnamed: 32' and 'id').
    - Converts the 'diagnosis' column to a binary format (1 for 'M', 0 for 'B').

    Args:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    # Drop unused columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map 'diagnosis' values: 'M' (Malignant) -> 1, 'B' (Benign) -> 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def normalize_data(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize the numerical features in the dataset.

    Args:
        data (pd.DataFrame): The dataset to normalize.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The normalized dataset as a NumPy array.
            - StandardScaler: The scaler object used for normalization.
    """
    # Create a StandardScaler instance to normalize the data
    scalar = StandardScaler()
    
    # Apply the scaler to the data and transform it
    data = scalar.fit_transform(data)

    return data, scalar