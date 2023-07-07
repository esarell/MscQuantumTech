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

-----------------------------------------------------------------------------------------------------------------------------------------

The data is stored in the file:


-----------------------------------------------------------------------------------------------------------------------------------------
Hyperparameters are stored:
Training.py

learning_rate = 0.01
batch_size = 25
epochs = 10

Epochs currently determined by "steps" in QCNN_run.py
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