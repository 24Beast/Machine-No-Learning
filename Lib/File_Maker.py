# Importing Libraries
import os
import sys

# Class Definition
class File_Creator():
    
    def __init__(self,user_name,data_type,args=""):        
        command = "cp Lib\\template.py "
        if(sys.platform[:3] == "win"):
            command = "copy Lib\\template.py "
        command = command + user_name + "\\main.py"
        os.system(command)
        self.user_name = user_name
        self.dtype = data_type
        self.args = args
    
    def write_file(self,x_cols=None,y_cols=None):
        if(self.dtype=="CSV"):
            file = open(self.user_name+"\\main.py","r+")
            lines = file.readlines()
            lines.insert(5,"from CSV_Data_Loader import Data_Loader\n")
            lines.insert(8,"data = Data_Loader("+str(self.args)+")\n")
            lines.insert(9,"X_cols = "+str(x_cols)+"\n")
            lines.insert(10,"y_col = "+str(y_cols)+"\n")
            lines.insert(11,"X,y = data.read(X_cols,y_col)\n")
            file.seek(0)
            file.writelines(lines)
            file.close()
        else:
            file = open(self.user_name+"\\main.py","r+")
            lines = file.readlines()
            lines.insert(5,"from Image_Data_Loader import Data_Loader\n")
            lines.insert(8,"data = Data_Loader("+str(self.args)+")\n")
            lines.insert(9,"X,y = data.read()\n")
            file.seek(0)
            file.writelines(lines)
            file.close()
        

# # Execution
# # user_name = sys.argv[1]
# user_name = "Bhanu_img"
# # csv = File_Creator(user_name,"CSV",{"fname":"abc.csv","clean":True})
# # csv.write_file(["A","B","C"],["D"])
# image = File_Creator(user_name,"Image",{"fname":"abc","shape":(120,120)})
# image.write_file()
