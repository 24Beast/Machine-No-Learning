from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from Simple_Models import Model_Creator as creator

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("image",type=str)
parser.add_argument("count",type=int)

class Main_Page(Resource):
    def __init__(self):
        pass
    
    def get(self):
        return {"Hello":"World"}
    
    def post(self):
        json_data = request.get_json(force = True)
        username = json_data("user_name")
        model = creator(json_data)
        model.get_new_model()
        model.save("Models",username)
        return {"Happy":"Noises"}

class User_Page(Resource):
    def __init__(self):
        pass
    
    def get(self,name):
        json_data = request.get_json(force = True)
        model = creator(json_data)
        try:
            model.load_model("Models/"+name+".sav")
        except FileNotFoundError:
            return {"Sad":"Noises"}
        return {"Model":model}
    
        
api.add_resource(Main_Page,"/")
api.add_resource(User_Page,"/name/<string:name>")

if __name__=="__main__":
    app.run(debug=True)