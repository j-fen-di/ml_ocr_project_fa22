from pathlib import Path
import os
import string
import shutil


def filterData():
    datasetPath = ""
    currPath = os.path.abspath(os.getcwd())
    datasetPath = os.path.join(currPath, "data", "dataset")
    dirlist = os.listdir(datasetPath)
    specialChars = ["alpha", "beta", "delta", "div", "forward_slash", "gamma", "geq",
                    "gt", "infty", "lambda", "leq", "lt", "mu", "neq", "phi", "pi", "pm",
                    "sigma", "theta", "times", "!", "(", ")", "+", "-", "=", "[", "]", "{", "}"]

    currChars = list(string.ascii_lowercase) + \
        list(string.digits) + specialChars

    for dirname in dirlist:
        if dirname.lower() not in currChars:
            if dirname.endswith('.DS_Store'):
                removePath = os.path.join(datasetPath, dirname)
                os.remove(removePath) 
            print("delete: ", dirname)
            removePath = os.path.join(datasetPath, dirname)
            shutil.rmtree(removePath)

filterData()
