# MscQuantumTech
This project focuses on using Quantum Convolution Nural Networks (QCNN) in order to determine signal for nosie for Gravitational Waves

Currently two students have looked into this uses pennylane as the main libary. This repostory is a continuation of there work.
-----------------------------------------------------------------------------------------------------------------------------------------

There are 6 main files used:

Benchmarking.py

QCNN_circuit.py

QCNN_run.py

Quantum_Data.py

sin_generator.py

Training.py

QCNN_test.py:
allow you to run previsously trained models and test them

Also the CNN.py file to benchmark the system
-----------------------------------------------------------------------------------------------------------------------------------------

The data is stored in the file:
qdata_1009.txt
This was generated using either the Quantum_Data.py or the sin_generator.py file

-----------------------------------------------------------------------------------------------------------------------------------------
Hyperparameters are stored:
Training.py

These where the inital
learning_rate = 0.01
batch_size = 25
epochs = 500

Epochs currently determined by "steps" in QCNN_run.py
Steps is not exactly the same as epochs, an epoch is when it runs through the whole data set the step is each step is just a different batch

For the current data file we are doing that means that 300 steps is equivalent to 10 epochs when the batch size is 25


Testing of hyper params
-----------------------------------------------------------------------------------------------------------------------------------------
How to run.

Run the QCNN_run.py file
Adjust the filename ect to make it work for you?

Set up make sure you have three folders set up,
Models, Results and Data to collect the information

Models: Holds files for the parameters
Results: Holds the cost function/loss output
Data: Holds the specific split up of test and training set data so you can test without testing on seen data

Models are pickeld versions of the paramaters at that point, file naming convention works by using model + step number + C + Cost + .pkl

The data that was randomly split is also saved in the Data folder with a date corresponding to each bit of data. This is also pickeld

Run the CNN File:
----------------------------------------------------
To test a model run QCNN_test.py file.
This will then promt you to laoad in a model file and its subsequent data.

This will then give you an accuracy for that against the models test set, which is unseen.