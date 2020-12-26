# Importing Necessary Libraries
import numpy as np
import pandas as pd

# Class Definition
class Data_Loader():
    
    def __init__(self,fname,clean=False):
        '''
        Parameters
        ----------
        fname : string
            filename for csv data, folder name for image classification.
        clean : Boolean/int, optional
            If CSV file needs to be cleaned. The default is False.

        Returns
        -------
        None.
        '''
        self.fname = fname
        self.clean = clean

    def read(self,features,target):
        '''
        Returns
        -------
        X : numpy array.
            Input features.
        y : numpy array
            Target function.
        '''        
        data = pd.read_csv(self.fname)
        print("Detected Headers are :")
        print([x for x in data])
        if(self.clean):
            for x in data:
                x[x.isnull()] = x.median()  
        X = data[features].values
        y = data[target].values
        return X,y
        
            