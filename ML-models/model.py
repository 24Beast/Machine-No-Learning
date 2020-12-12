import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten 
from math import floor, ceil
class modelToBeImported():
	def createModel(self):
		self.model = Sequential()
		self.model.add(Conv2D(6, 3, (1, 1), valid, input_shape = (12,12, 3), activation = None))
		self.model.add(Conv2D(12, 3, (1, 1), valid, activation = None))
		self.model.add(Conv2D(24, 3, (1, 1), valid, activation = None))
		self.model.add(Flatten())
		self.model.add(Dense(432, activation = None))
		self.model.add(Dense(216, activation = None))
		self.model.add(Dense(108, activation = None))
		self.model.add(Dense(54, activation = None))
		self.model.add(Dense(3, activation = softmax))
