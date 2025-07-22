############################################################# Imports #############################################################
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

############################################################# Setup Data #############################################################

inputTrainData = np.genfromtxt('datasets/DPA_200MHz/train_input.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
outputTrainData = np.genfromtxt('datasets/DPA_200MHz/train_output.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

inputTestData = np.genfromtxt('datasets/DPA_200MHz/test_input.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
outputTestData = np.genfromtxt('datasets/DPA_200MHz/test_output.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

input_I_train = inputTrainData[:, 0]
input_Q_train = inputTrainData[:, 1]
input_I_train = input_I_train.astype(np.float64)
input_Q_train = input_Q_train.astype(np.float64)

output_I_train = outputTrainData[:, 0]
output_Q_train = outputTrainData[:, 1]
output_I_train = output_I_train.astype(np.float64)
output_Q_train = output_Q_train.astype(np.float64)

input_I_test = inputTestData[:, 0]
input_Q_test = inputTestData[:, 1]
input_I_test = input_I_test.astype(np.float64)
input_Q_test = input_Q_test.astype(np.float64)

output_I_test = outputTestData[:, 0]
output_Q_test = outputTestData[:, 1]
output_I_test = output_I_test.astype(np.float64)
output_Q_test = output_Q_test.astype(np.float64)

def plot_data(input_I, input_Q, output_I, output_Q, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(output_I, output_Q, color='red', label='Output Data')
    plt.scatter(input_I, input_Q, color='blue', label='Input Data')
    plt.title(title)
    plt.xlabel('I Component')
    plt.ylabel('Q Component')
    plt.legend()
    plt.grid()
    plt.show()

plot_data(input_I_train, input_Q_train, output_I_train, output_Q_train, 'Training Data')
plot_data(input_I_test, input_Q_test, output_I_test, output_Q_test, 'Testing Data')
t = np.linspace(0, 23040, 23040)
plot_data(t, input_I_train*np.cos(t) + input_Q_train*np.sin(t), t, output_I_train*np.cos(t) + output_Q_train*np.sin(t), 'Signals')

