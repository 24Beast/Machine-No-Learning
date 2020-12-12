import numpy as np

class Data_Loader():
    
    def __init__(self,fname,data_form,clean=False,shape=None):
        '''
        Parameters
        ----------
        fname : string
            filename for csv data, folder name for image classification.
        data_form : string
            "Image" or "CSV" to select type of input data.
        clean : Boolean/int, optional
            If CSV file needs to be cleaned. The default is False.
        shape : tuple/list, optional
            Desired shape of input images. The default is None.

        Returns
        -------
        None.
        '''
        self.fname = fname
        self.dtype = data_form
        self.clean = None
        self.shape = None
        if(self.dtype=="CSV"):
            self.clean = clean
        else:
            self.shape = tuple(shape)
            
    def reader(self):
        '''
        Returns
        -------
        X : numpy array.
            Input features/images.
        y : numpy array
            Target function.
        '''
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
                    X[count] = cv2.resize(cv2.imread(self.fname+files[i]+img),self.shape)
                    y[count] = i
                    count+=1
            return X,y
        else:
            import pandas as pd
            data = pd.read_csv(self.fname)
            print("Detected Headers are :")
            print([x for x in data])
            if(self.clean):
                for x in data:
                    x[x.isnull()] = x.median()  
            choice = -1
            features = []
            while choice<0:
                feature = input("Enter a feature column (Enter 0 to exit)")
                if(feature=="0"):
                    choice=1
                while feature not in data:
                    feature = input("\rEnter Correct Target Column :")
                if(feature not in features):
                    features.append(feature)
            target = input("Enter Target column : ")
            while target not in data:
                target = input("\rEnter Correct Target Column :")
            X = data[features].values
            y = data[[target]].values
            return X,y
            
            