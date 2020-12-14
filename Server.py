# Importing Required Libraries
import os
import sys
from Lib.Modeling_Functions import *
from Lib.File_Maker import File_Creator
from File_Maker import File_Creator
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
        username = json_data("user_name")
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
            os.system("del /q "+username)
        else:
            return {"Sad Noises" : "User Not Found."}
        json_data = request.get_json(force = True)
        model = model_map[str(json_data("model"))]
        data_type = str(json_data("data_type"))
        args = json_data("args")
        command = "cp "
        if(sys.platform[:3] == "win"):
            command = "copy "
        if(data_type == "Image"):
            command += image + name + "\\Image_Data_Loader.py"
        else:
            command += csv + name + "\\CSV_Data_Loader.py"
        os.system(command)
        model(name)
        f = File_Creator(name,data_type)
        f.write_file()
        return {"Happy":"Noises"}
        
api.add_resource(Main_Page,"/")
api.add_resource(User_Page,"/name/<string:name>")

if __name__=="__main__":
    app.run(debug=True)