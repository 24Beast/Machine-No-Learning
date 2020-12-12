#Importing Necessary Libraries
from sklearn.ensemble import RandomForestClassifier

#Class Definition

class modelToBeImported():

	def __init__(self):
		self.model = None

	def createModel(self):
		self.model = RandomForestClassifier({'x': 1, 'y': 2})

	def fit(self,X,y):
		self.model.fit(X,y)

	def predict(self,X):
		return self.model.predict(X)
