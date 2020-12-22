# Importing Libraries
import os
import sys
from Lib.Modeling_Functions import *
from Lib.File_Maker import File_Creator

# Intializing Variables
csv = "Lib\\CSV_Data_Loader.py "
image = "Lib\\Image_Data_Loader.py "
username = "Bhanu"
data_type = "Image"

# Execution
if (os.path.isdir(username)):
    os.system("del /q "+username)
else:
    os.mkdir(username)
command = "cp "
if(sys.platform[:3] == "win"):
    command = "copy "
if(data_type == "Image"):
    command += image + username + "\\Image_Data_Loader.py"
else:
    command += csv + username + "\\CSV_Data_Loader.py"
os.system(command)
kNearestNeighborsClassificationConstructor(username)
f = File_Creator(username,data_type)
f.write_file()