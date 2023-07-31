import pickle
import torch.nn as nn
import torch
import numpy as np

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def accuracy_test(predictions, labels):
    acc = 0
    for (p,l) in zip(predictions, labels):
        if p[0] >= p[1]:
            pred = 0 
        else:
            pred = 1

        if pred == l:
            acc = acc + 1
    acc = acc / len(labels)
    return acc
def loadModel(filename):
    paramfile = open(filename, 'rb')     
    params = pickle.load(paramfile)
    paramfile.close()
    return params

def testModel(CNN,filename):
    input_size = 256
    datafile = open(filename, 'rb')     
    currentdata = pickle.load(datafile)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = currentdata
    datafile.close()

    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    X_test_torch.resize_(len(X_test), 1, input_size)
    Y_pred = CNN(X_test_torch).detach().numpy()
    accuracy = accuracy_test(Y_pred, Y_test)
    print("Accuracy:" + str(accuracy))


if __name__ == "__main__":
    name = input("Enter file name of the model you would like to load: ")
    model =loadModel(name)
    #print(model)
    oldData = input("Enter file name of data: ")
    testModel(model,oldData)

    #accuracy = accuracy_test(Y_pred, Y_val)