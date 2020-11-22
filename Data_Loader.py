import numpy as np

class Data_Loader():
    
    def __init__(self,opt):
        self.fname = opt["fname"]
        self.dtype = opt["data-form"] #Images vs csv
        self.clean = None
        self.shape = None
        if(self.dtype=="CSV"):
            self.clean = opt["clean"]
        else:
            self.shape = opt["shape"]
            
    def reader(self):
        if(self.dtype == "Image"):
            import cv2,os
            files = os.listdir(self.fname)
            num_files = 0
            for file in files:
                num_files += len(os.listdir(self.fname+file))
            X = np.zeros((num_files,*self.shape))
            y = np.zeros(num_files)
            count = 0
            for i in range(len(files)):
                imgs = os.listdir(self.fname+files[i])
                for img in imgs:
                    X[count] = cv2.imread(self.fname+files[i]+img)
                    y[count] = i
                    count+=1
            return X,y
        else:
            import pandas as pd
            data = pd.read_csv(self.fname)
            X = data.iloc[:,:-1].values
            y = data.iolc[:,-1].values
            if(self.clean):
                #Implementation pending....
                pass
            return X,y
            
            