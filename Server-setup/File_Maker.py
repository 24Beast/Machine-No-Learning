# Importing Libraries
import os
import sys

# Class Definition
class File_Maker():
    
    def __init__(self,user_name,model,X_cols,y_cols,args=None):        
        command = "cp template.py "
        if(sys.platform[:3] == "win"):
            command = "copy template.py "
        command = command + user_name +".py"
        os.system(command)
        self.user_name = user_name
        self.model = model
        self.args = args
        self.mod = 26
        self.x = X_cols
        self.y = y_cols
        self.model_name = ""
        self.lib = ""
    
    def write_file(self):
        self.get_model_info()
        file = open(user_name+".py","r+")
        lines = file.readlines()
        lines.insert(7,"x = "+str(self.x)+"\n")
        lines.insert(8,"y = "+str(self.y)+"\n")
        lines.insert(11,"args = "+str(args)+"\n")
        lines.insert(self.mod,"from "+self.lib+" import "+self.model_name+"\n")
        self.mod+=1
        lines.insert(self.mod,"regressor = "+self.model_name+"(args)")
        self.mod+=1
        file.seek(0)
        file.writelines(lines)
        file.close()
    
    def get_model_info(self):
        if(self.model == "Linear_Regressor"):
            self.model_name = "LinearRegression"
            self.lib = "sklearn.linear_model"
        elif(self.model == "Random-Forest-Regressor"):
            self.model_name = "RandomForestRegressor"
            self.lib = "sklearn.ensemble"
        elif(self.model == "KNN-Classifier"):
            self.model_name = "KNeighborsClassifier"
            self.lib = "sklearn.neighbors"
        elif(self.model == "Random-Forest-Classifier"):
            self.model_name = "RandomForestClassifier"
            self.lib = "sklearn.ensemble"
        

# Execution
user_name = sys.argv[1]
model = sys.argv[2]
# model = "Random-Forest-Regressor"
# user_name = "Bhanu"
args = {"x":1,"y":2,"z":3}
x_cols = ["a","b"]
y_cols = ["c"]
z = File_Maker(user_name, model, x_cols, y_cols)
z.write_file()