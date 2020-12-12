# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import modelToBeImported

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X_col = ['a', 'b']
y_col = ['c']
X = dataset[X_col].values
y = dataset[y_col].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating and Fitting the Model to the dataset
# model.py will have the wrapper model class: modelToBeImported who's name will be same regardless of which model it is
model = modelToBeImported()
model.fit(X_train,y_train)

# Predicting a new result
y_pred = model.predict(X_test)
