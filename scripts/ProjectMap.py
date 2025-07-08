import os

# Simple class that keeps track of the paths to various directories in the project
# i.e. "The path to the /scripts directory is XXXX"
class ProjectMap():
    def __init__(self):
        self.scriptsDirectory= os.path.dirname(os.path.abspath(__file__))
        self.projectRootDirectory= os.path.dirname(self.scriptsDirectory)
        self.dataDirectory= os.path.join(self.projectRootDirectory, 'data')

    def __str__(self):
        out= f"Root: {self.projectRootDirectory}\nScripts: {self.scriptsDirectory}\nData: {self.dataDirectory}"
        return out