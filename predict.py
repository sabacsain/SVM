import os, pickle, traceback
from pathlib import Path


def get_filepath() -> Path:
    try:
        # Get the current working directory
        current_path = os.getcwd()

        # Define the directory path
        directory_path = Path(current_path)

        # Concatenate the directory path and dataset location
        dataset_path = directory_path / 'svm_model.pkl'

    # Check for errors
    except Exception as e:
        print('An error occured at get_filepath()')
        print("Error: ", e)

        # Print the traceback information
        traceback.print_exc()

        return None
    
    # Successful extraction of filepath
    else:
        print('Filepath extracted successfully')
        return dataset_path


def load_model(filepath) -> object:
    try:
        with open(filepath, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

    except Exception as e:
        print('An error occured at load_model()')
        print('Error: ', e)

        # Print traceback information
        traceback.print_exc()

        return None

    else:
        print('Dataset loaded successfully')
        return loaded_model


def test_model(loaded_model) -> int:
    try:
        test_input = [2.9543,1.076,0.64577,0.89394]
        predicted_label = loaded_model.predict([test_input])
    
    except Exception as e:
        print('An error occured: ', e)
        return None
        
    else:
        print('Model predicted successfully')
        return predicted_label


def display_prediction(prediction) -> None:
    match prediction:
        case 0:
            print('Prediction: Bank Note is Genuine') 
        case 1:  
            print('Prediction: Bank Note is Fake')
        case _:
            print('Unexpected Output occured')

    
if __name__ == "__main__":

    # Get filepath of the model
    filepath = get_filepath()

    # Exit if there is an error getting the filepath
    if filepath is None:
        exit(1)

    # Load the model
    model = load_model(filepath)

    # Exit if there is an error loading the model
    if model is None:
        exit(1)

    # Predict using SVM
    prediction = test_model(model)

    # Exit if there is an error predicting the model
    if prediction is None:
        exit(1)

    # Display the result
    display_prediction(prediction)



