from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# from ucimlrepo import fetch_ucirepo 
  
import os, pickle, traceback
import pandas as pd
import numpy as np





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


# def fetch_dataset() -> bool:
#     try:
#         # fetch dataset 
#         dataset = fetch_ucirepo(id=267) 
#         print(type(dataset))

#     except Exception as e:
#         print('An error occured', e)
#         return None

#     else:
#         print('Data fetched successfully')
#         return dataset
    

def create_model(directory_path, dataset_path, df):
    try:
        # Split the data into features and labels
        X_train = df.drop(df.columns[-1], axis=1)
        y_train = df[df.columns[-1]]

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4)

        # # Convert text data into numerical features using CountVectorizer
        # vectorizer = CountVectorizer()
        # X_train_vectorized = vectorizer.fit_transform(X_train.apply(lambda x: ' '.join(x), axis=1))
        # X_test_vectorized = vectorizer.transform(X_test.apply(lambda x: ' '.join(x), axis=1))

        # Create SVM classifier
        svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

        # Train the classifier
        svm_classifier.fit(X_train.values, y_train)

        # Predict the labels of the test data
        y_pred = svm_classifier.predict(X_test)

        # Save the model to a file using pickle
        save_path =  directory_path / 'svm_model.pkl'
        with open(save_path, 'wb') as model_file:
            pickle.dump(svm_classifier, model_file)
        print('SVM Model created successfully')

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    except Exception as e:
        print('An error occured: ', e)

        # Print the traceback information
        traceback.print_exc()

        return None

    return 0



if __name__ == "__main__":
    filepath = get_filepath()

    # Exit if there is an error getting the file
    if filepath is None:
        exit(1)

    df = load_dataset(filepath['dataset_path'])

    # Exit if there is an error reading the data
    if df is None:
        exit(1)

    create_model(filepath['directory_path'], filepath['dataset_path'], df)

    
    

    


    
        

