from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
import pandas as pd


def get_filepath() -> Path:
    try:
        # Define the dataset path
        dataset_path = Path.cwd() / 'dataset.csv'
        
    except Exception as e:
        print("An error occurred:", e)
        return None
    
    else:
        print('Filepath Extraction: Success')
        return dataset_path


def load_dataset(dataset_path) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_path)
        
    except Exception as e:
        print("An error occurred:", e)
        return None
    
    else:
        print("Load Dataset: Success")
        return df


def create_model(df):
    try:
        # Split the data into features and labels
        X_train = df.drop('Label', axis=1)
        y_train = df['Label']

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4)

        # Create SVM classifier
        svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

        # Train the classifier
        svm_classifier.fit(X_train.values, y_train)

        # Predict the labels of the test data
        y_pred = svm_classifier.predict(X_test)

         # Save the model to a file using pickle
        with open('svm_model.pkl', 'wb') as model_file:
            pickle.dump(svm_classifier, model_file)
        print('Model Creaton: Success')

        # Display Performance Matrix
        print(classification_report(y_test, y_pred)) 

    except Exception as e:
        print('An error occured: ', e)


# Main Driver
if __name__ == "__main__":

    # Get filepath
    dataset_path = get_filepath()

    # Exit if there is an error getting the file
    if dataset_path is None:
        exit(1)

    # Load dataset
    df = load_dataset(dataset_path)

    # Exit if there is an error reading the dataset
    if df is None:
        exit(1)

    # Create the SVM Model
    create_model(df)

    
    

    


    
        

