import numpy as np

class Model_Creator():
    
    def __init__(self,opt):
        self.model_type = opt["model_type"]
        self.model_args = opt["model_args"]
        self.model = None
        
    def get_new_model(self):
        if(self.model_type.split("_")[-1]=="Regressor"):
            if(self.model_type == "Linear-Regressor"):
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression(**self.model_args)
            elif(self.model_type == "Support-Vector-Regressor"):
                import sklearn.svm as SVR
                self.model = SVR(**self.model_args)
            elif(self.model_type == "Decision-Tree-Regressor"):
                from sklearn.tree import DecisionTreeRegressor as DTR
                self.model = DTR(**self.model_args)
            elif(self.model_type == "Random-Forest-Regressor"):
                from sklearn.ensemble import RandomForestRegressor as RFR
                self.model = RFR(**self.model_args)
        else:
            if(self.model_type == "Logistic-Regression-Classifier"):
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(**self.model_args)
            elif(self.model_type == "KNN-Classifier"):
                from sklearn.neighbors import KNeighborsClassifier as KNN
                self.model = KNN(**self.model_args)
            elif(self.model_type == "Support-Vector-Classifier"):
                import sklearn.svm as SVC
                self.model = SVC(**self.model_args)
            elif(self.model_type == "Naive-Bayes-Classifier"):
                from sklearn.naive_bayes import GNB
                self.model = GNB(**self.model_args)
            elif(self.model_type == "Decision-Tree-Classifier"):
                from sklearn.tree import DecisionTreeClassifier as DTC
                self.model = DTC(**self.model_args)
            elif(self.model_type == "Random-Forest-Classifier"):
                from sklearn.ensemble import RandomForestClassifier as RFC
                self.model = RFC(**self.model_args)
    
    def train(self,X,y,val_size=0.2):
        from sklearn.model_selection import train_test_split
        X_train,X_val,y_train,y_val = train_test_split(X,y, test_size=val_size)
        self.model.fit(X_train,y_train)
        if(self.model_type.split("_")[-1]=="Regressor"):
            from sklearn.metrics import mean_squared_error as mse
            from sklearn.metrics import r2_score as r2
            y_pred_train = self.model.predict(X_train)
            print("Training Scores:")
            print("MSE : "+str(mse(y_train,y_pred_train)))
            print("R-Squared-Score : "+str(r2(y_train,y_pred_train)))
            if(val_size!=0):
                y_pred_val = self.model.predict(X_val)
                print("Validation Scores:")
                print("MSE : "+str(mse(y_val,y_pred_val)))
                print("R-Squared-Score : "+str(r2(y_val,y_pred_val)))
        else:
            from sklearn.metrics import classification_report as cr
            y_pred_train = self.model.predict(X_train)
            print("Training Scores:")
            print("MSE : "+str(cr(y_train,y_pred_train)))
            if(val_size!=0):
                y_pred_val = self.model.predict(X_val)
                print("Validation Scores:")
                print("MSE : "+str(cr(y_val,y_pred_val)))
    
    def predict(self,X):
        return self.model.predict(X)
    
    def save_model(self,save_dir="",model_name="Sample_Model"):
        import pickle
        fname = save_dir+"/"+model_name+".sav"
        with open(fname,"wb") as f:
            pickle.dump(self.model,f)
    
    def load_model(self,model_loc):
        if(model_loc==None):
            print("Model Not Found.")
        else:
            import pickle
            with open(model_loc,"rb") as f:
                self.model = pickle.load(f)
    
    