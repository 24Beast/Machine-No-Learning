# Importing Required Libraries
import os
import cv2
import numpy as np


class Data_Loader():
    
    def __init__(self,fname,shape=(240,320)):
        '''
        Parameters
        ----------
        fname : string
            filename for csv data, folder name for image classification.
        shape : tuple/list, optional
            Desired shape of input images. The default is None.

        Returns
        -------
        None.
        '''
        self.fname = fname
        self.shape = shape

    def reader(self):
        '''
        Returns
        -------
        X : numpy array.
            Input images.
        y : numpy array
            Target function.
        '''
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

            