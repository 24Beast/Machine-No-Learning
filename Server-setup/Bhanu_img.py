# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import modelToBeImported
from Image_Data_Loader import Data_Loader

# Importing the dataset
data = Data_Loader({'fname': 'abc', 'shape': (120, 120)})
X,y = data.read()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating and Fitting the Model to the dataset
# model.py will have the wrapper model class: modelToBeImported who's name will be same regardless of which model it is
model = modelToBeImported()
model.fit(X_train,y_train)

# Predicting a new result
y_pred = model.predict(X_test)
