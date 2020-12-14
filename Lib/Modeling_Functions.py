# Collection of Wrapper Functions to create models (both Neural Nets and traditional Libraries) used by the Machine-No-Learning API

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from math import ceil

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def createDenseNetwork(numberOfLayers, numberOfInputNodes, numberOfOutputNodes, typeOfModel = "Regression", hiddenLayerNodes = None, activationFunctions = None):
    # Linear Neural Network / Multi Layer Perceptron Network
    
    # numOfLayers: number of layers(compulsory, including input and output layers, >=2)
    # numOfInputNodes: number of input variables(compulsory)
    # numOfOutputNodes: number of output variables(compulsory)
    # typeOfModel: Regression or Classification (Default: Regression)
    # hiddenLayerNodes: array of number of nodes in each hidden layer (Default: increases uniformly for half of the layers and then decreases uniformly for the latter half)
    # activationFunctions: list of activation functions excluding input layer(Default: None for hidden layers and for output layer: Softmax(Classification) or None(Regression))
    
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


def create2DCNNNetwork(numOfConvLayers, numOfDenseLayers, inputShape, numOfOutCats, typeOfModel = "Classification", numOfFilters = None, kernelSizes = None, strides = None, padding = "valid", activationFunctions = None):

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



# Sklearn models =>

def create_model_file(lib, model_name, args):
#     self.get_model_info()
    file = open("model.py","w")
    lines = ["#Importing Necessary Libraries\n",
             "from "+lib+" import "+model_name+"\n",
             "\n#Class Definition\n\nclass modelToBeImported():\n\n",
             "\tdef createModel(self):\n"]
    lines.append("\t\tself.model = "+model_name+"("+args+")\n\n")
    lines.append("\tdef fit(self,X,y):\n")
    lines.append("\t\tself.model.fit(X,y)\n\n")
    lines.append("\tdef predict(self,X):\n")
    lines.append("\t\treturn self.model.predict(X)\n")
    file.seek(0)
    file.writelines(lines)
    file.close()
    
    
def linearRegressionConstructor(includeIntercept = True, normalizeData = False, copyX = True):
    # includeIntercepts: if the data is centered around 0, we don't need this, basically handles bias
    # normalizeData: normalisation involves subtracting mean and dividing by l2-norm
    # copyX: in case of normalising the data, the original may or may not be overwritten, this creates a copy of the data
    
    model = LinearRegression(fit_intercept = includeIntercept, normalize = normalizeData, copy_X = copyX)
    args = "fit_intercept = {}, normalize = {}, copy_X = {}".format(includeIntercept, normalizeData, copyX)
    create_model_file("sklearn.linear_model","LinearRegression",args)
    return model

def randomForestRegressionConstructor(numTrees = 100, maxDepth = None, numForSplit = 2, numForLeaf = 1):
    # numTrees: number of trees in the forest
    # maxDepth: max depth/ height of each tree in the forest
    # numForSplit: min number of samples for splitting an internal node (non leaf)
    # numForLeaf: min number of samples for being a leaf node, changing from 1 can help smooth the model
    
    model = RandomForestRegressor(n_estimators = numTrees, max_depth = maxDepth, min_samples_split = numForSplit, min_samples_leaf = numForLeaf)
    args = "n_estimators = {}, max_depth = {}, min_samples_split = {}, min_samples_leaf = {}".format(numTrees, maxDepth, numForSplit, numForLeaf)
    create_model_file("sklearn.ensemble","RandomForestRegressor",args)
    return model

def randomForestClassificationConstructor(numTrees = 100, estFunc = "gini", maxDepth = None, numForSplit = 2, numForLeaf = 1):
    # numTrees: number of trees in the forest
    # estFunc: the function to measure the quality of a split (responsible for deciding when to split) => gini or entropy
    # maxDepth: max depth/ height of each tree in the forest
    # numForSplit: min number of samples for splitting an internal node (non leaf)
    # numForLeaf: min number of samples for being a leaf node, changing from 1 can help smooth the model
    
    model = RandomForestClassifier(n_estimators = numTrees, criterion = estFunc, max_depth = maxDepth, min_samples_split = numForSplit, min_samples_leaf = numForLeaf)
    args = "n_estimators = {}, criterion = \"{}\", max_depth = {}, min_samples_split = {}, min_samples_leaf = {}".format(numTrees, estFunc, maxDepth, numForSplit, numForLeaf)
    create_model_file("sklearn.ensemble","RandomForestClassifier",args)
    return model

def kNearestNeighborsClassifier(numNeighbors = 5, weighedByDistance = False, distType = 2):
    # numNeighbors: number of nearest neighbors considered for estimation
    # weighedByDistance: whether or not influence in decision varies according to distance
    # distType: power parameter for minkowski metric: 1=> l1(manhattan distance) or 2=> l2(euclidian distance)
    
    model = KNeighborsClassifier(n_neighbors = numNeighbors, weights = ("distance" if weighedByDistance else "uniform"), p = distType)
    args = "n_neighbors = {}, weights = \"{}\", p = {}".format(numNeighbors, "distance" if weighedByDistance else "uniform", distType)
    create_model_file("sklearn.neighbors","KNeighborsClassifier",args)
    return model
        