# Importing Libraries
import os
import sys

# Class Definition
class File_Creator():
    
    def __init__(self,user_name,X_cols,y_cols,args=None):        
        command = "cp template.py "
        if(sys.platform[:3] == "win"):
            command = "copy template.py "
        command = command + user_name +".py"
        os.system(command)
        self.user_name = user_name
        self.x = X_cols
        self.y = y_cols
    
    def write_file(self):
        file = open(user_name+".py","r+")
        lines = file.readlines()
        lines.insert(8,"X_col = "+str(self.x)+"\n")
        lines.insert(9,"y_col = "+str(self.y)+"\n")
        file.seek(0)
        file.writelines(lines)
        file.close()
    
        

# Execution
user_name = sys.argv[1]
# user_name = "Bhanu"
x_cols = ["a","b"]
y_cols = ["c"]
z = File_Creator(user_name, x_cols, y_cols)
z.write_file()