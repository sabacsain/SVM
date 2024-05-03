import pickle
from pathlib import Path


def get_filepath() -> Path:
    try:
        # Define the dataset path
        dataset_path = Path.cwd() / 'svm_model.pkl'

    # Check for errors
    except Exception as e:
        print("Error: ", e)
        return None
    
    # Successful extraction of filepath
    else:
        print('Filepath Extraction: Success')
        return dataset_path


def load_model(filepath) -> object:
    try:
        with open(filepath, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
    
    except Exception as e:
        print('An error occurred: ', e)
        return None
    
    else:
        print('Load Model: Success')
        return loaded_model


def test_model(loaded_model) -> int:
    try:
        # Change test_input as necessary
        test_input = [2.9543,1.076,0.64577,0.89394] 
        predicted_label = loaded_model.predict([test_input])
    
    except Exception as e:
        print('An error occured: ', e)
        return None
        
    else:
        print('Classification: Success')
        return predicted_label


def display_prediction(prediction) -> None:
    if prediction == 0:
        print('Prediction: Bank Note is Genuine') 
    elif prediction == 1:  
        print('Prediction: Bank Note is Fake')
    else:
        print('Unexpected Output occurred')


# Main driver function 
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

    # Predict using the Model
    prediction = test_model(model)

    # Exit if there is an error predicting the model
    if prediction is None:
        exit(1)

    # Display the result
    display_prediction(prediction)



