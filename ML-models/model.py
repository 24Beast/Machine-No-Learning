import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten 
from math import floor, ceil
class modelToBeImported():
	def createModel(self):
		self.model.add(Dense(10, input_shape = (5,), activation = None))
		self.model.add(Dense(20, activation = None))
		self.model.add(Dense(40, activation = None))
		self.model.add(Dense(20, activation = None))
		self.model.add(Dense(10, activation = None))
		self.model.add(Dense(1, activation = None))
