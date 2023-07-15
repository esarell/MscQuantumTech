import pickle
import QCNN_circuit
import Benchmarking
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

def testParameters(params,filename):
    print("Paramaters loaded")
    datafile = open(filename, 'rb')     
    currentdata = pickle.load(datafile)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = currentdata
    datafile.close()
    U = 'U_6'
    U_params = 10
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X_test]
    accuracy = Benchmarking.accuracy_test(predictions, Y_test, True)
    print("Accuracy for " + U + " Amplitude :" + str(accuracy))



if __name__ == "__main__":
    name = input("Enter file name of the model you would like to load: ")
    params =loadParams(name)
    print(params)
    oldData = input("Enter file name of data: ")
    testParameters(params,oldData)
