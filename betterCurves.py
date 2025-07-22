####################################################################################### Imports #######################################################################################
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rfpaModel import PowerAmplifierModel
from rotatedinverseRFPAModel import MLP

####################################################################################### Constants #######################################################################################
filterInputThreshold = 3
filterOutputHighValue = 3
paGain = 3.2

####################################################################################### Set up ML #######################################################################################

# Set up RFPA Model
model_path = 'power_amplifier_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = PowerAmplifierModel()
loaded_model.to(device)
print(f"Using device: {device}")
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval()
print("Model loaded successfully for inference.")

# Set up the inverse RFPA Model
inverse_model_path = 'inverse_pa_model.pth'
inverse_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inverse_loaded_model = PowerAmplifierModel()
inverse_loaded_model.to(inverse_device)
print(f"Using device: {inverse_device}")
inverse_loaded_model.load_state_dict(torch.load(inverse_model_path))
inverse_loaded_model.to(inverse_device)
inverse_loaded_model.eval()
print("Inverse Model loaded successfully for inference.")

#################################################################### Function Definitions ####################################################################

# Saturation Curve (for RFPA Model)
def RFPA(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(device)
        with torch.no_grad():
            predicted_output.append(loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

# Inverse Saturation Curve (for inverse RFPA Model)
def inverseRFPA(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(inverse_device)
        with torch.no_grad():
            predicted_output.append(inverse_loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

def multiplyByGain(val):
    return (val * paGain * inverseRFPA(np.array([2]))[0])

# FIR Filter (for filtering out high values)
def filter(val):
    return (val*np.heaviside(-1*(val-filterInputThreshold), 1) + filterOutputHighValue*np.heaviside(val-filterInputThreshold, 1))

####################################################################################### Signal Testing #######################################################################################

t = np.linspace(0, 1000, 1000)

inputSignal = 0.2 * np.sin(0.1 * t) + 0.08 * np.sin(0.08 * t) + 0.2 * (np.random.randn(len(t)) / 10.0) + 0.5

# Desired Signal
desiredSignal = inputSignal * paGain

# Amplification without DPD
outputWithoutDPD = RFPA(inputSignal)

# Amplification with DPD with Proposed Pipeline (Input -> Filter -> Inverse Block -> Mulitply by Gain -> PA -> Output)
signalAfterFilter = filter(inputSignal)
signalAfterInverse = inverseRFPA(signalAfterFilter)
signalAfterMultiply = multiplyByGain(signalAfterInverse)
outputWithDPD = RFPA(signalAfterMultiply)


# Graph everything
# plt.plot(t, inputSignal, label='Input Signal')
# plt.plot(t, signalAfterFilter, label='Post-Filter Signal')
# plt.plot(t, signalAfterInverse, label='Post-Inverse Signal')
# plt.plot(t, signalAfterMultiply, label='Post-Multiply Signal')
# plt.plot(t, outputWithDPD, label='Output with DPD')
# plt.plot(t, outputWithoutDPD, label='Output without DPD')
# plt.plot(t, desiredSignal, label='Desired Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Visualization of DPD Process and Comparison with non-DPD method')
# plt.legend()
# plt.show()

####################################################################################### Tranformation Testing #######################################################################################

x = np.linspace(0, 3, 1000)

# Tranformation Without DPD
transformationWithoutDPD = RFPA(x)

# Tranformation With DPD
transformationWithDPD = RFPA(multiplyByGain(inverseRFPA(filter(x))))

# Desired Tranformation
transformationDesired = paGain*x*np.heaviside(-1*(x-filterInputThreshold), 1) + 3.1*np.heaviside(x-(filterInputThreshold), 0)

# Graph everything
# plt.plot(x, transformationDesired, label='Desired')
# plt.plot(x, transformationWithoutDPD, label='Without DPD')
# plt.plot(x, transformationWithDPD, label='With DPD')
# plt.xlabel('Input Amplitude')
# plt.ylabel('Output Amplitude')
# plt.title('Visualization of Output Power vs Input Power for with DPD and without')
# plt.legend()
# plt.show()

####################################################################################### Tianyi Edits #######################################################################################

width = 1
plt.style.use('dark_background')
plt.rcParams["font.family"] = "sans-serif"
plt.plot(t, inputSignal, label='Input Signal', color='cyan', linewidth = width)
plt.plot(t, RFPA(inputSignal), label='Output Without DPD', color='red', linewidth = width)
plt.plot(t, RFPA(inverseRFPA(paGain * inputSignal)), label='Output With DPD', color='violet', linewidth = width)
plt.plot(t, inputSignal*paGain, label='Desired Output', color='orange', linewidth = width)
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.title('Comparison of Amplification With and Without DPD')
plt.legend()
plt.show()

plt.plot(x, RFPA(x), label='PA Model')
plt.plot(x, inverseRFPA(x), label='Inverse PA Model')
plt.plot(x, filter(paGain*x), label='Desired')
plt.plot(x, filter(RFPA(inverseRFPA(paGain * x))), label='Overall System')
plt.xlabel('Input Amplitude')
plt.ylabel('Output Amplitude')
plt.title('Visualization of Output Power vs Input Power')
plt.legend()
plt.show()

modifiedSignal = RFPA(inverseRFPA(paGain * inputSignal))
desiredSignal = inputSignal * paGain
regularAmplifiedSignal = RFPA(inputSignal)
mse = np.mean((modifiedSignal - desiredSignal)**2)
mseRegular = np.mean((regularAmplifiedSignal - desiredSignal)**2)
print("MSE with DPD: " + str(mse))
print("MSE without DPD: " + str(mseRegular))

