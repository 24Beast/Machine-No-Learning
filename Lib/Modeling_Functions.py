#!/usr/bin/env python
# coding: utf-8

# In[4]:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from math import ceil


# In[34]:


# Linear Neural Network / Multi Layer Perceptron Network


# inputs to the construction function:
# number of layers(compulsory, including input and output layers, >=2)
# number of input variables(compulsory)
# number of output variables(compulsory)
# type of model: Regression or Classification (Default: Regression)
# array of number of nodes in each hidden layer (Default: increases uniformly for half of the layers and then decreases uniformly for the latter half)
# activation functions excluding input layer(Default: None for hidden layers and for output layer: Softmax(Classification) or None(Regression))
# 

def createDenseNetwork(numberOfLayers, numberOfInputNodes, numberOfOutputNodes, typeOfModel = "Regression", hiddenLayerNodes = None, activationFunctions = None):
    
    # Writing the starting includes in the model.py file
    f = open("model.py",'w')
    f.write("import tensorflow as tf \nimport numpy as np \nfrom tensorflow.keras import Sequential \nfrom tensorflow.keras.layers import Dense, Conv2D, Flatten \nfrom math import floor, ceil\n")    
    f.write("class modelToBeImported():\n\tdef createModel(self):\n")
    
    layersAdded = 0
    
    # model initialisation
    model = Sequential()
    f.write("\t\tself.model = Sequential()\n")
    
    # Handling stupid exceptions
    
    if activationFunctions == None:
        
        activationFunctions = [None]*(numberOfLayers-1)
    
    if numberOfLayers<2:
        
        print("Dude seriously??!!!!!! number of layers is less than 2, not possible...")
        # throw exception
        return model
    
    elif (int(numberOfInputNodes) != numberOfInputNodes):
        
        print("Bro seriously?! fractional nodes?!")
        # throw exception
        return model
    
    # no hidden layers
    
    elif numberOfLayers == 2:
        
        f.write("\t\tself.")
        model.add(Dense(numberOfOutputNodes, input_shape = (numberOfInputNodes,), activation = activationFunctions[0]))
        f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(numberOfOutputNodes, numberOfInputNodes, None if (activationFunctions[0] == None) else ("\"{}\"".format(activationFunctions[0]))))
        model.summary()
        layersAdded += 1
        return model
    
    else:
        
        numberOfHiddenLayers = numberOfLayers - 2
        
        if hiddenLayerNodes == None:
            # If the number of nodes in each layer are not specified
            
            threshold = 512
            
            # Input layer
            f.write("\t\tself.")
            model.add(Dense(min(numberOfInputNodes*2,threshold), input_shape = (numberOfInputNodes,), activation = activationFunctions[layersAdded]))
            f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(min(numberOfInputNodes*2,threshold),numberOfInputNodes,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
            
            layersAdded += 1
            
            # print("numberOfHiddenLayers: \t",ceil(numberOfHiddenLayers))
            # print("Ceil value: \t",ceil(numberOfHiddenLayers/2))
            
            # Increasing the number of nodes for first half of the network
            
            for layer in range(1,ceil(numberOfHiddenLayers/2)):
                nodeCount = min(numberOfInputNodes*(2**(layer+1)),threshold)
                f.write("\t\tself.")
                model.add(Dense(nodeCount, activation = activationFunctions[layersAdded]))
                f.write("model.add(Dense({}, activation = {}))\n".format(nodeCount,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
                print(nodeCount)
                
            # Decreasing the number of nodes for the second half of the network
            
            for layer in range(ceil(numberOfHiddenLayers/2),numberOfHiddenLayers):
                nodeCount = int(nodeCount/2)
                nodeCount = max(nodeCount,numberOfOutputNodes)
                f.write("\t\tself.")
                model.add(Dense(nodeCount, activation = activationFunctions[layersAdded]))
                f.write("model.add(Dense({}, activation = {}))\n".format(nodeCount,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
        
        else:
            # Number of nodes is specified as a list of integers
            
            # input layer 
            f.write("\t\tself.")
            model.add(Dense(hiddenLayerNodes[0], input_shape = (numberOfInputNodes,), activation = activationFunctions[layersAdded]))
            f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(hiddenLayerNodes[0], numberOfInputNodes, None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
            layersAdded += 1
            
            # Hidden layers
            for hiddenLayer in range(1,len(hiddenLayerNodes)):
                f.write("\t\tself.")
                model.add(Dense(hiddenLayerNodes[hiddenLayer], activation = activationFunctions[layersAdded]))
                f.write("model.add(Dense({}, activation = {}))\n".format(hiddenLayerNodes[hiddenLayer], None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
                
        # Last Layer: regression or classification
        
        if typeOfModel == "Regression":
            
            f.write("\t\tself.")
            model.add(Dense(numberOfOutputNodes, activation = activationFunctions[layersAdded]))
            f.write("model.add(Dense({}, activation = {}))\n".format(numberOfOutputNodes,activationFunctions[layersAdded]))
        
        elif typeOfModel == "Classification":
            
            f.write("\t\tself.")
            model.add(Dense(numberOfOutputNodes, activation = "softmax" if (activationFunctions[layersAdded] == None) else activationFunctions[layersAdded]))
            f.write("model.add(Dense({}, activation = {}))\n".format(numberOfOutputNodes, "\"softmax\"" if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))

        model.summary()
        f.close()
        
        return model    



# In[28]:


# Standard 2D-CNN network

# structure: upscaling CNN layers followed by flatten layer, followed by downscaling dense layers
# 
# inputs to the construction function:
# number of convlayers(compulsory) these do not include the input layer i.e. can be >=1
# number of denselayers(compulsory) these do not include the output layer i.e. can be >=1
# input_shape((height, width, channels),compulsory)
# number of output categories(compulsory)
# type of model: Regression or Classification (Default: Classification)
# array of number of filters in each conv layer (Default: doubles the number of filters in each conv layer)
# array of kernel sizes (Default: 3x3)
# array of strides (Default: 1x1)
# padding (Default: Valid (i.e. No padding))
# 
# 
# activation functions(Default: None for hidden layers and for output layer: Softmax(Classification) or None(Regression))
# 
# 

def create2DCNNNetwork(numOfConvLayers, numOfDenseLayers, inputShape, numOfOutCats, typeOfModel = "Classification", numOfFilters = None, kernelSizes = None, strides = None, padding = "valid", activationFunctions = None):
    
    # Writing starting includes in the model.py file
    
    f = open("model.py",'w')
    f.write("import tensorflow as tf \nimport numpy as np \nfrom tensorflow.keras import Sequential \nfrom tensorflow.keras.layers import Dense, Conv2D, Flatten \nfrom math import floor, ceil\n")    
    f.write("class modelToBeImported():\n\tdef createModel(self):\n")
    
    # Initialising model
    model = Sequential()
    f.write("\t\tself.model = Sequential()\n")
    
    # Initialising the number of filters, kernel size, stride, activation function per conv layer list in case it is set to None
    
    if numOfFilters == None:
        fs = inputShape[-1]
        numOfFilters = []
        
        for i in range(numOfConvLayers):
            numOfFilters.append(2*fs)
            fs = 2*fs
    
    else:
        
        if len(numOfFilters != numOfConvLayers):
            # throw error
            return
        
    if kernelSizes == None:
        
        kernelSizes = [3]*numOfConvLayers
    
    if strides == None:
        
        strides = [(1,1)]*numOfConvLayers
    
    if activationFunctions == None:
        
        activationFunctions = [None]*(numOfConvLayers+numOfDenseLayers)
        
        if typeOfModel == "Classification":
            activationFunctions.append("softmax")
    
    # Adding Convolution Layers: default pattern follows doubling number of filters
    f.write("\t\tself.")
    model.add(Conv2D(numOfFilters[0], kernelSizes[0], strides[0], padding, input_shape = (inputShape[-3],inputShape[-2], inputShape[-1]), activation = activationFunctions[0]))
    f.write("model.add(Conv2D({}, {}, {}, \"{}\", input_shape = ({},{}, {}), activation = {}))\n".format(numOfFilters[0], kernelSizes[0], strides[0], padding, inputShape[-3],inputShape[-2], inputShape[-1], None if (activationFunctions[0] == None) else ("\"{}\"".format(activationFunctions[0]))))
    
    for i in range(1,numOfConvLayers):
        f.write("\t\tself.")
        model.add(Conv2D(numOfFilters[i], kernelSizes[i], strides[i], padding, activation = activationFunctions[i]))
        f.write("model.add(Conv2D({}, {}, {}, \"{}\", activation = {}))\n".format(numOfFilters[i], kernelSizes[i], strides[i], padding, None if (activationFunctions[i] == None) else ("\"{}\"".format(activationFunctions[i]))))
    
    # Adding flattening later to change [height, width, filters] to [height x width x filters]
    model.add(Flatten())
    f.write("\t\tself.model.add(Flatten())\n")
    
    # Adding Dense Layers: default pattern follows number of nodes = max(outCategories, previousLayerNodes/2)
    nodeCount = model.output.shape[-1]
    nodeCountArray = []
    nodeCount = max(int(nodeCount/2),numOfOutCats)
    nodeCountArray.append(nodeCount)
    
    for i in range(numOfDenseLayers):
        nodeCount = max(int(nodeCount/2),numOfOutCats)
        nodeCountArray.append(nodeCount)
    # print(activationFunctions)
    
    for i in range(numOfDenseLayers):
        f.write("\t\tself.")
        model.add(Dense(nodeCountArray[i], activation = activationFunctions[i+numOfConvLayers]))
        f.write("model.add(Dense({}, activation = {}))\n".format(nodeCountArray[i], None if (activationFunctions[i+numOfConvLayers] == None) else ("\"{}\"".format(activationFunctions[i+numOfConvLayers]))))
    
    f.write("\t\tself.")
    model.add(Dense(numOfOutCats, activation = activationFunctions[-1]))
    f.write("model.add(Dense({}, activation = {}))\n".format(numOfOutCats, None if (activationFunctions[-1] == None) else ("\"{}\"".format(activationFunctions[-1]))))
    
    model.summary()
    f.close()
    
    return model


# In[ ]:


# SK_Model_Creator : Model file creator for sklearn based 
class SK_Model_Creator():
    
    def __init__(self,model_type,model_args,file_name="model"):
        self.model_type = model_type
        self.model_args = model_args
        self.file = file_name
        self.model_name = None
        self.lib = None
        
    def get_model_info(self):
        if(self.model_type == "Linear_Regressor"):
            self.model_name = "LinearRegression"
            self.lib = "sklearn.linear_model"
        elif(self.model_type == "Random-Forest-Regressor"):
            self.model_name = "RandomForestRegressor"
            self.lib = "sklearn.ensemble"
        elif(self.model_type == "KNN-Classifier"):
            self.model_name = "KNeighborsClassifier"
            self.lib = "sklearn.neighbors"
        elif(self.model_type == "Random-Forest-Classifier"):
            self.model_name = "RandomForestClassifier"
            self.lib = "sklearn.ensemble"
    
    def create_model_file(self):
        self.get_model_info()
        file = open(self.file+".py","w")
        lines = ["#Importing Necessary Libraries\n",
                 "from "+self.lib+" import "+self.model_name+"\n",
                 "\n#Class Definition\n\nclass modelToBeImported():\n\n",
                 "\tdef __init__(self):\n\t\tself.model = None\n\n",
                 "\tdef createModel(self):\n"]
        lines.append("\t\tself.model = "+self.model_name+"("+str(self.model_args)+")\n\n")
        lines.append("\tdef fit(self,X,y):\n")
        lines.append("\t\tself.model.fit(X,y)\n\n")
        lines.append("\tdef predict(self,X):\n")
        lines.append("\t\treturn self.model.predict(X)\n")
        file.seek(0)
        file.writelines(lines)
        file.close()
        

