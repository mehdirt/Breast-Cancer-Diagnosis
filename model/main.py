import pickle
from utils.data_utils import get_data, clean_data
from utils.model_utils import create_model

def main() -> None:
    """
    Main function that orchestrates the data processing, model training, 
    and saving of the trained model and scalar.

    Steps:
    1. Load and clean the data from a CSV file.
    2. Train a model using the cleaned data.
    3. Save the trained model and scalar to files for later use.
    """

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


# Entry point for the script.
if __name__ == '__main__':
    main()