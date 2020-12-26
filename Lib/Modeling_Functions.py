# Collection of Wrapper Functions to create models (both Neural Nets and traditional Libraries) used by the Machine-No-Learning API

from math import ceil


def createDenseNetwork(user_name,numberOfLayers, numberOfInputNodes, numberOfOutputNodes, typeOfModel = "Regression", hiddenLayerNodes = None, activationFunctions = None):
    # Linear Neural Network / Multi Layer Perceptron Network
    
    # user_name: Folder where file is to be generated
    # numOfLayers: number of layers(compulsory, including input and output layers, >=2)
    # numOfInputNodes: number of input variables(compulsory)
    # numOfOutputNodes: number of output variables(compulsory)
    # typeOfModel: Regression or Classification (Default: Regression)
    # hiddenLayerNodes: array of number of nodes in each hidden layer (Default: increases uniformly for half of the layers and then decreases uniformly for the latter half)
    # activationFunctions: list of activation functions excluding input layer(Default: None for hidden layers and for output layer: Softmax(Classification) or None(Regression))
    
    # Writing the starting includes in the model.py file
    f = open(user_name+"\\model.py",'w')
    f.write("import tensorflow as tf \nimport numpy as np \nfrom tensorflow.keras import Sequential \nfrom tensorflow.keras.layers import Dense, Conv2D, Flatten \nfrom math import floor, ceil\n")    
    f.write("class modelToBeImported():\n\tdef createModel(self):\n")
    
    layersAdded = 0
    
    # model initialisation
    f.write("\t\tself.model = Sequential()\n")
    
    # Handling stupid exceptions
    
    if activationFunctions == None:
        
        activationFunctions = [None]*(numberOfLayers-1)
    
    if numberOfLayers<2:
        
        print("Dude seriously??!!!!!! number of layers is less than 2, not possible...")
        # throw exception
    
    elif (int(numberOfInputNodes) != numberOfInputNodes):
        
        print("Bro seriously?! fractional nodes?!")
        # throw exception
    
    # no hidden layers
    
    elif numberOfLayers == 2:
        
        f.write("\t\tself.")
        f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(numberOfOutputNodes, numberOfInputNodes, None if (activationFunctions[0] == None) else ("\"{}\"".format(activationFunctions[0]))))
        layersAdded += 1
    
    else:
        
        numberOfHiddenLayers = numberOfLayers - 2
        
        if hiddenLayerNodes == None:
            # If the number of nodes in each layer are not specified
            
            threshold = 512
            
            # Input layer
            f.write("\t\tself.")
            f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(min(numberOfInputNodes*2,threshold),numberOfInputNodes,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
            
            layersAdded += 1
            
            # print("numberOfHiddenLayers: \t",ceil(numberOfHiddenLayers))
            # print("Ceil value: \t",ceil(numberOfHiddenLayers/2))
            
            # Increasing the number of nodes for first half of the network
            
            for layer in range(1,ceil(numberOfHiddenLayers/2)):
                nodeCount = min(numberOfInputNodes*(2**(layer+1)),threshold)
                f.write("\t\tself.")
                f.write("model.add(Dense({}, activation = {}))\n".format(nodeCount,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
                print(nodeCount)
                
            # Decreasing the number of nodes for the second half of the network
            
            for layer in range(ceil(numberOfHiddenLayers/2),numberOfHiddenLayers):
                nodeCount = int(nodeCount/2)
                nodeCount = max(nodeCount,numberOfOutputNodes)
                f.write("\t\tself.")
                f.write("model.add(Dense({}, activation = {}))\n".format(nodeCount,None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
        
        else:
            # Number of nodes is specified as a list of integers
            
            # input layer 
            f.write("\t\tself.")
            f.write("model.add(Dense({}, input_shape = ({},), activation = {}))\n".format(hiddenLayerNodes[0], numberOfInputNodes, None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
            layersAdded += 1
            
            # Hidden layers
            for hiddenLayer in range(1,len(hiddenLayerNodes)):
                f.write("\t\tself.")
                f.write("model.add(Dense({}, activation = {}))\n".format(hiddenLayerNodes[hiddenLayer], None if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))
                layersAdded += 1
                
        # Last Layer: regression or classification
        
        if typeOfModel == "Regression":
            
            f.write("\t\tself.")
            f.write("model.add(Dense({}, activation = {}))\n".format(numberOfOutputNodes,activationFunctions[layersAdded]))
        
        elif typeOfModel == "Classification":
            
            f.write("\t\tself.")
            f.write("model.add(Dense({}, activation = {}))\n".format(numberOfOutputNodes, "\"softmax\"" if (activationFunctions[layersAdded] == None) else ("\"{}\"".format(activationFunctions[layersAdded]))))

        f.close()


def create2DCNNNetwork(user_name,numOfConvLayers, numOfDenseLayers, inputShape, numOfOutCats, typeOfModel = "Classification", numOfFilters = None, kernelSizes = None, strides = None, padding = "valid", activationFunctions = None):

    # Standard 2D-CNN network
    
    # user_name: Folder where file is to be generated
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
    
    f = open(user_name+"\\model.py",'w')
    f.write("import tensorflow as tf \nimport numpy as np \nfrom tensorflow.keras import Sequential \nfrom tensorflow.keras.layers import Dense, Conv2D, Flatten \nfrom math import floor, ceil\n")    
    f.write("class modelToBeImported():\n\tdef createModel(self):\n")
    
    # Initialising model
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
    f.write("model.add(Conv2D({}, {}, {}, \"{}\", input_shape = ({},{}, {}), activation = {}))\n".format(numOfFilters[0], kernelSizes[0], strides[0], padding, inputShape[-3],inputShape[-2], inputShape[-1], None if (activationFunctions[0] == None) else ("\"{}\"".format(activationFunctions[0]))))
    
    for i in range(1,numOfConvLayers):
        f.write("\t\tself.")
        f.write("model.add(Conv2D({}, {}, {}, \"{}\", activation = {}))\n".format(numOfFilters[i], kernelSizes[i], strides[i], padding, None if (activationFunctions[i] == None) else ("\"{}\"".format(activationFunctions[i]))))
    
    # Adding flattening later to change [height, width, filters] to [height x width x filters]
    f.write("\t\tself.model.add(Flatten())\n")
    
    # Adding Dense Layers: default pattern follows number of nodes = max(outCategories, previousLayerNodes/2)
    nodeCountArray = []
    nodeCount = max(int(nodeCount/2),numOfOutCats)
    nodeCountArray.append(nodeCount)
    
    for i in range(numOfDenseLayers):
        nodeCount = max(int(nodeCount/2),numOfOutCats)
        nodeCountArray.append(nodeCount)
    # print(activationFunctions)
    
    for i in range(numOfDenseLayers):
        f.write("\t\tself.")
        f.write("model.add(Dense({}, activation = {}))\n".format(nodeCountArray[i], None if (activationFunctions[i+numOfConvLayers] == None) else ("\"{}\"".format(activationFunctions[i+numOfConvLayers]))))
    
    f.write("\t\tself.")
    f.write("model.add(Dense({}, activation = {}))\n".format(numOfOutCats, None if (activationFunctions[-1] == None) else ("\"{}\"".format(activationFunctions[-1]))))
    
    f.close()



# Sklearn models =>

def create_model_file(user_name,lib, model_name, args):
#     self.get_model_info()
    file = open(user_name+"\\model.py","w")
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
    
    
def linearRegressionConstructor(user_name,includeIntercept = True, normalizeData = False, copyX = True):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    includeIntercept : Bool, optional
        If the data is centered around 0, we don't need this, basically handles bias. The default is True.
    normalizeData : Bool, optional
        Normalisation involves subtracting mean and dividing by l2-norm. The default is False.
    copyX : Bool, optional
        In case of normalising the data, the original may or may not be overwritten, this creates a copy of the data. The default is True.

    Returns
    -------
    None.

    """
    args = "fit_intercept = {}, normalize = {}, copy_X = {}".format(includeIntercept, normalizeData, copyX)
    create_model_file(user_name,"sklearn.linear_model","LinearRegression",args)


def randomForestRegressionConstructor(user_name,numTrees = 100, maxDepth = None, numForSplit = 2, numForLeaf = 1):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    numTrees : int, optional
        Number of trees in the forest. The default is 100.
    maxDepth : int, optional
        Max depth/ height of each tree in the forest. The default is None.
    numForSplit : int, optional
        Min number of samples for splitting an internal node (non leaf). The default is 2.
    numForLeaf : int, optional
        Min number of samples for being a leaf node, changing from 1 can help smooth the model. The default is 1.

    Returns
    -------
    None.
    
    """
    
    args = "n_estimators = {}, max_depth = {}, min_samples_split = {}, min_samples_leaf = {}".format(numTrees, maxDepth, numForSplit, numForLeaf)
    create_model_file(user_name,"sklearn.ensemble","RandomForestRegressor",args)


def supportVectorRegressorConstructor(user_name,kernel = "rbf", degree = 3, gamma = "scale", epsilon = 0.1):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated. 
    kernel : string, optional
        Kernel to be used by SVM Regressor. The default is "rbf".
    degree : int, optional
        Degree of polynomial kernel. The default is 3.
    gamma : string, optional
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        if ‘auto’, uses 1 / n_features.. The default is "scale".
    epsilon : float, optional
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.. The default is 0.1.
        
    Returns
    -------
    None.


    """
    args = "kernel = {}, degree = {}, gamma = {}, epsilon ={}".format(kernel, degree,gamma,epsilon)
    create_model_file(user_name,"sklearn.svm","SVR",args)


def randomForestClassificationConstructor(user_name,numTrees = 100, estFunc = "gini", maxDepth = None, numForSplit = 2, numForLeaf = 1):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    numTrees : int, optional
        number of trees in the forest. The default is 100.
    estFunc : string, optional
        the function to measure the quality of a split (responsible for deciding when to split) => gini or entropy. The default is "gini".
    maxDepth : int, optional
        Max depth/ height of each tree in the forest. The default is None.
    numForSplit : int, optional
        Min number of samples for splitting an internal node (non leaf). The default is 2.
    numForLeaf : int, optional
        Min number of samples for being a leaf node, changing from 1 can help smooth the model. The default is 1.

    Returns
    -------
    None.

    """
    
    args = "n_estimators = {}, criterion = \"{}\", max_depth = {}, min_samples_split = {}, min_samples_leaf = {}".format(numTrees, estFunc, maxDepth, numForSplit, numForLeaf)
    create_model_file(user_name,"sklearn.ensemble","RandomForestClassifier",args)


def kNearestNeighborsClassificationConstructor(user_name,numNeighbors = 5, weighedByDistance = False, distType = 2):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    numNeighbors : int, optional
        Number of nearest neighbors considered for estimation. The default is 5.
    weighedByDistance : Bool, optional
        Whether or not influence in decision varies according to distance. The default is False.
    distType : int, optional
        power parameter for minkowski metric: 1=> l1(manhattan distance) or 2=> l2(euclidian distance). The default is 2.

    Returns
    -------
    None.

    """
    
    args = "n_neighbors = {}, weights = \"{}\", p = {}".format(numNeighbors, "distance" if weighedByDistance else "uniform", distType)
    create_model_file(user_name,"sklearn.neighbors","KNeighborsClassifier",args)


def supportVectorClassifierConstructor(user_name,kernel = "rbf", degree = 3, gamma = "scale", epsilon = 0.1, class_weigth = None):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated. 
    kernel : string, optional
        Kernel to be used by SVM Regressor. The default is "rbf".
    degree : int, optional
        Degree of polynomial kernel. The default is 3.
    gamma : string, optional
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        if ‘auto’, uses 1 / n_features.. The default is "scale".
    epsilon : float, optional
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.. The default is 0.1.
    class_weigth : dict, optional
        Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. 
        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). 
        The default is None.

    Returns
    -------
    None.
    """
    
    args = "kernel = {}, degree = {}, gamma = {}, epsilon ={}, class_weight = {}".format(kernel, degree, gamma, epsilon, class_weigth)
    create_model_file(user_name,"sklearn.svm","SVC",args)


def naiveBayesClassifierConstructor(user_name,priors = None, var_smoothing = 1e-9): 
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    priors : array-like of shape(n_classes,), optional
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data. The default is None.
    var_smoothing : float, optional
        Portion of the largest variance of all features that is added to variances for calculation stability. The default is 1e-9.

    Returns
    -------
    None.
    """
    args = "priors = {}, var_smoothing = {}".format(priors,var_smoothing)
    create_model_file(user_name,"sklearn.naive_bayes","GaussianNB",args)


def logisticRegressionConstructor(user_name,penalty = "l2"):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, optional
        Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. 
        ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.. The default is "l2".

    Returns
    -------
    None.

    """
    args = "penalty = {}".format(penalty)
    create_model_file(user_name,"sklearn.linear_model","LogisticRegression",args)


def kMeansClusteringConstructor(user_name, n_clusters = 8, init = "k-means++", n_init = 10, max_iter = 300):
    """
    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    n_clusters : int, optional
        The number of clusters to form as well as the number of centroids to generate. The default is 8.
    init : string/array like object, optional
        Method for initialization:
        
        ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
        
        ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.
        
        If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        
        If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization. 
        
        The default is "k-means++".
    n_init : int, optional
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. The default is 10.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm for a single run. The default is 300.

    Returns
    -------
    None.
    """
    args = "n_clusters = {},init = {},n_init = {},max_iter = {}".format(n_clusters,init,n_init,max_iter)
    create_model_file(user_name,"sklearn.cluster","KMeans",args)

def agglomerativeClusteringConstructor(user_name,n_clusters = 2, affinity = "euclidean", linkage = "ward", distance_threshold = None):
    """

    Parameters
    ----------
    user_name : string
        Folder where file is to be generated.
    n_clusters : int, optional
        The number of clusters to find. It must be None if distance_threshold is not None. The default is 2.
    affinity : string, optional
        Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. 
        If linkage is “ward”, only “euclidean” is accepted. If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
        The default is "euclidean".
    linkage : float, optional
        Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
        
        ‘ward’ minimizes the variance of the clusters being merged.
        
        ‘average’ uses the average of the distances of each observation of the two sets.
        
        ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.
        
        ‘single’ uses the minimum of the distances between all observations of the two sets.. The default is "ward".
    distance_threshold : float, optional
        The linkage distance threshold above which, clusters will not be merged. If not None, n_clusters must be None and compute_full_tree must be True.. The default is None.

    Returns
    -------
    None.

    """
    args = "n_clusters = {}, affinity = {}, linkage = {},distance_threshold={}".format(n_clusters,affinity,linkage,distance_threshold)
    create_model_file(user_name, "sklearn.cluster", "AgglomerativeClustering", args)