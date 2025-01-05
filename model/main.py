import pickle
from utils.data_utils import get_data, clean_data
from utils.model_utils import create_model

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