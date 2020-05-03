from os import path
directory = path.dirname(__file__)
filepath = path.abspath(path.join(directory,"..","Classification"))
print(filepath)
