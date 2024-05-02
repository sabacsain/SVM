from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 

import os, pickle, traceback
import pandas as pd


def get_filepath() -> Path:
    try:
        # Get the current working directory
        current_path = os.getcwd()

        # Define the directory path
        directory_path = Path(current_path)

        # Concatenate the directory path and dataset location
        dataset_path = directory_path / 'dataset.txt'

    # Check for errors
    except Exception as e:
        print("An error occured: ", e)
        return None
    
    # Successful extraction of filepath
    else:
        print('Filepaths extracted successfully')
        return {
            'directory_path'    :directory_path,
            'dataset_path'      :dataset_path        
        }


def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)

    # Check for errors
    except Exception as e:
        print("An error occured: ", e)
        return None

    # Successful reading of dataset
    else:
        print("Dataset loaded successfully")
        return df


def create_model(directory_path, dataset_path, df):
    try:
        # Split the data into features and labels
        X_train = df.drop('Label', axis=1)
        y_train = df['Label']

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4)

        # Create SVM classifier
        svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

        # Train the classifier
        svm_classifier.fit(X_train, y_train)

        # Predict the labels of the test data
        y_pred = svm_classifier.predict(X_test)

        # Save the model to a file using pickle
        save_path =  directory_path / 'svm_model.pkl'
        with open(save_path, 'wb') as model_file:
            pickle.dump(svm_classifier, model_file)
        print('SVM Model created successfully')

        # Display Performance Matrix
        print(classification_report(y_test, y_pred)) 

    except Exception as e:
        print('An error occured: ', e)

        # Print the traceback information
        traceback.print_exc()

        return None

    return 0


# Main Driver
if __name__ == "__main__":
    # Get filepath
    filepath = get_filepath()

    # Exit if there is an error getting the file
    if filepath is None:
        exit(1)

    df = load_dataset(filepath['dataset_path'])

    # Exit if there is an error reading the data
    if df is None:
        exit(1)

    create_model(filepath['directory_path'], filepath['dataset_path'], df)

    
    

    


    
        

