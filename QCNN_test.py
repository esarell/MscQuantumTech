import pickle

'''
Loads in a pickeled file and then unpickels it
Specifically does this for the paramaters
ARG: file name
Returns: list of paramaters
'''
def loadParams(filename):
    paramfile = open(filename, 'rb')     
    params = pickle.load(paramfile)
    paramfile.close()
    return params



if __name__ == "__main__":
    name = input("Enter file name of the model you would like to load:")
    print(loadParams(name))