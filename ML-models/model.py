import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Conv2D, Flatten 
from math import floor, ceil
class modelToBeImported():
	def createModel(self):
		self.model = Sequential()
		self.model.add(Dense(1, input_shape = (5,), activation = None))
