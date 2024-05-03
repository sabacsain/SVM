import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 


# Save dataset
cancer = load_breast_cancer() 

# Feature columns
df_feat = pd.DataFrame(cancer['data'], 
                       columns = cancer['feature_names']) 
  
# Label column
df_target = pd.DataFrame(cancer['target'],  
                     columns =['Cancer']) 
                       
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(df_feat, 
                                                        np.ravel(df_target), 
                                                        test_size = 0.30, 
                                                        random_state = 101) 

# Train the model
model = SVC() 
model.fit(X_train, y_train) 
  
# Display results
predictions = model.predict(X_test) 
print(classification_report(y_test, predictions)) 

exit(0)


#####################################################
# Hyperparameter Tuning
#####################################################

from sklearn.model_selection import GridSearchCV 
  
# Defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
# Train the model with Hyperparameter Tuning
model = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
model.fit(X_train, y_train) 

# Display results
grid_predictions = model.predict(X_test) 
print(classification_report(y_test, grid_predictions)) 
