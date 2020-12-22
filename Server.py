# Importing Required Libraries
import os
import sys
from Lib.Modeling_Functions import *
from Lib.File_Maker import File_Creator
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

# Initialization
app = Flask(__name__)
api = Api(app)
csv = "Lib\\CSV_Data_Loader.py "
image = "Lib\\Image_Data_Loader.py "
model_map = {"ANN" : createDenseNetwork,
             "CNN" : create2DCNNNetwork,
             "Linear_Reg" : linearRegressionConstructor,
             "Random_Forest_Reg" :randomForestRegressionConstructor,
             "KNN_Class" : kNearestNeighborsClassificationConstructor,
             "Random_Forest_Class" : randomForestClassificationConstructor}

class Main_Page(Resource):
    def __init__(self):
        pass
    
    def get(self):
        return {"Hello":"World"}
    
    def post(self):
        json_data = request.get_json(force = True)
        username = json_data["user_name"]
        if (os.path.isdir(username)):
            os.system("del /q "+username)
        else:
            os.mkdir(username)
        return {"Happy":"Noises"}

class User_Page(Resource):
    def __init__(self):
        pass
    
    def get(self,name):
        if (os.path.isdir(name)):
            return {"Len" : len(os.listdir(name))}
        else:
            return {"Len" : -1}
            
    def post(self,name):
        if (os.path.isdir(name)):
            os.system("del /q "+name)
        else:
            return {"Sad Noises" : "User Not Found."}
        json_data = request.get_json(force = True)
        model = model_map[str(json_data["model"])]  # Model selection from those available in model_map
        data_type = str(json_data["data_type"])     # Image or CSV
        args = json_data["args"]                    # Arguements for the model  
        fname = json_data["fname"]                  # File to be read (Will be converted to readable url for Google Drive Shareable Link )
        if(fname.startswith("https://drive.google.com/file/d")):
            fname = path = 'https://drive.google.com/uc?export=download&id='+fname.split('/')[-2]
        X_cols = json_data["X"]                     # Feature Column No. (Index starts at 0)
        y_cols = json_data["y"]                     # Target Column/s.
        command = "cp "
        if(sys.platform[:3] == "win"):
            command = "copy "
        if(data_type == "Image"):
            command += image + name + "\\Image_Data_Loader.py"
        else:
            command += csv + name + "\\CSV_Data_Loader.py"
        os.system(command)
        model(name,**args)
        f = File_Creator(name,data_type,fname)
        f.write_file(X_cols,y_cols)
        return {"Happy":"Noises"}
        
api.add_resource(Main_Page,"/")
api.add_resource(User_Page,"/name/<string:name>")

if __name__=="__main__":
    app.run(debug=True)