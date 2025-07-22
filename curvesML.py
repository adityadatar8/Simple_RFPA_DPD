####################################################################################### Imports #######################################################################################
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rfpaModel import PowerAmplifierModel

####################################################################################### Constants #######################################################################################
threshold = 1.2
highValue = 3
degrees = 20
sampling_rate = 200000000 # in Hz

################################################################################## Set up the RFPA Model ##################################################################################
model_path = 'power_amplifier_model.pth'

# Check for GPU availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
loaded_model.to(device)
print(f"Using device: {device}")
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval() # Set to evaluation mode after loading
print("Model loaded successfully for inference.")

############################################################################# Set up the inverse RFPA Model #############################################################################
inverse_model_path = 'inverse_pa_model.pth'

# Check for GPU availability and move model to GPU if available
inverse_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inverse_loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
inverse_loaded_model.to(inverse_device)
print(f"Using device: {inverse_device}")
inverse_loaded_model.load_state_dict(torch.load(inverse_model_path))
inverse_loaded_model.to(inverse_device)
inverse_loaded_model.eval() # Set to evaluation mode after loading
print("Inverse Model loaded successfully for inference.")

############################################################################# Set up the rotated inverse RFPA Model #############################################################################
rotated_inverse_model_path = 'rotated_inverse_pa_model.pth'

# Check for GPU availability and move model to GPU if available
rotated_inverse_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rotated_inverse_loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
rotated_inverse_loaded_model.to(rotated_inverse_device)
print(f"Using device: {rotated_inverse_device}")
rotated_inverse_loaded_model.load_state_dict(torch.load(rotated_inverse_model_path))
rotated_inverse_loaded_model.to(rotated_inverse_device)
rotated_inverse_loaded_model.eval() # Set to evaluation mode after loading
print("Rotated Inverse Model loaded successfully for inference.")

#################################################################### Define the saturation and inverse saturation curves ####################################################################
x = np.linspace(0, 3, 1000)

def saturationCurve(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(device)
        with torch.no_grad():
            predicted_output.append(loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output
def inverseSaturationCurve(val):
    # reflection = [[0, 1], [1, 0]]
    # val2 = [x, saturationCurve(val)]
    # result = np.dot(reflection, val2)
    # return result[1]
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(inverse_device)
        with torch.no_grad():
            predicted_output.append(inverse_loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

f = saturationCurve(x)
g = inverseSaturationCurve(x)

######################################################################## Rotate the inverse saturation curve ########################################################################
theta = np.deg2rad(degrees)
rotation = [[np.cos(theta), -1*np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]]
def rotate(xArr, yArr):
    return np.dot(rotation, [xArr, yArr])
rotatedInverseSaturation = np.dot(rotation, [x, g])

def rotatedInverseSaturationCurve(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(rotated_inverse_device)
        with torch.no_grad():
            predicted_output.append(rotated_inverse_loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

###################################################### Cascade the rotated inverse saturation system with the saturation system ######################################################

# Possibly need to filter out the end of the inverse saturation curve
def filter(val):
    return (val*np.heaviside(-1*(x-threshold), 1) + highValue*np.heaviside(x-threshold, 1))

linearizedOut = saturationCurve(rotatedInverseSaturation[1])
linearizedIn = filter(rotatedInverseSaturation[0])

################################################################################## Plot everything ##################################################################################
plt.plot(x, f, label='PA Model Output')
plt.plot(x, g, label='Inverse PA Model Output')
plt.plot(rotatedInverseSaturation[0], rotatedInverseSaturation[1], label='Rotated Inverse PA Model Output')
plt.plot(linearizedIn, linearizedOut, label='Cascaded Output')
plt.legend()
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Cascading Rotated Inverse PA Model with PA Model')
plt.grid(True)
plt.xlim(0, 1.5)
plt.show()

############################################################################## Plot in Frequency Domain ##############################################################################

def plotInFrequencyDomain(xArray, yArray, plotLabel):
    fft_output = np.fft.fft(yArray)
    frequencies = np.fft.fftfreq(len(yArray), d=1/sampling_rate)

    # Take the magnitude and consider only the positive frequencies
    magnitude = np.abs(fft_output)
    positive_frequencies_mask = frequencies >= 0
    positive_frequencies = frequencies[positive_frequencies_mask]
    positive_magnitude = magnitude[positive_frequencies_mask]

    # Optional: Convert magnitude to dB
    magnitude_db = 20 * np.log10(positive_magnitude + 1e-9) # Add small value to avoid log(0)

    # 3. Plot the spectrum
    plt.plot(positive_frequencies, magnitude_db, label=plotLabel)
    plt.title('Spectrum Analyzer Output')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)

plotInFrequencyDomain(linearizedIn, linearizedOut, 'Cascaded Output')
plotInFrequencyDomain(x, f, 'PA Output')
plotInFrequencyDomain(x, g, 'Inverse PA Output')
plotInFrequencyDomain(rotatedInverseSaturation[0], rotatedInverseSaturation[1], 'Rotated Inverse PA Output')
plt.legend()
plt.show()


############################################################################## Test Amplification ##############################################################################
# 1. Generate a sample time-domain signal (e.g., a sine wave with noise)
sampling_rate = 1000  # Hz
duration = 1          # seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
frequency1 = 50       # Hz
frequency2 = 120      # Hz
amplitude1 = 0.1
amplitude2 = 0.08
signal = amplitude1 * np.sin(2 * np.pi * frequency1 * t) + amplitude2 * np.sin(2 * np.pi * frequency2 * t) + 0.2 * np.random.randn(len(t)) + 0.5 # Add some noise and DC offset

modifiedSignal = rotatedInverseSaturationCurve(signal)
modifiedSignal = saturationCurve(modifiedSignal)
#modifiedSignal[0] = filter(modifiedSignal[0])

desiredSignal = signal*2.4
regularAmplifiedSignal = saturationCurve(signal)

mse = np.mean((modifiedSignal - desiredSignal)**2)
mseRegular = np.mean((regularAmplifiedSignal - desiredSignal)**2)
print("MSE with DPD: " + str(mse))
print("MSE without DPD: " + str(mseRegular))

plt.plot(t, modifiedSignal, label='Amplified w/ DPD')
plt.plot(t, regularAmplifiedSignal, label='Amplified w/o DPD')
plt.plot(t, signal, label='Original Signal')
plt.plot(t, desiredSignal, label='Desired Signal')
plt.title('Amplification of a signal with Deep Learning-based DPD')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.legend()
plt.show()

plt.plot(t, regularAmplifiedSignal, label='Amplified w/o DPD')
plt.plot(t, signal, label='Original Signal')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Amplification of a signal without DPD')
plt.legend()
plt.show()

plt.plot(t, modifiedSignal, label='Amplified w/ DPD')
plt.plot(t, signal, label='Original Signal')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Amplification of a signal with Deep Learning-based DPD')
plt.legend()
plt.show()

plt.plot(t, regularAmplifiedSignal, label='Amplified w/o DPD')
plt.plot(t, desiredSignal, label='Desired Signal')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Amplification of a signal without DPD')
plt.legend()
plt.show()

plt.plot(t, modifiedSignal, label='Amplified w/ DPD')
plt.plot(t, desiredSignal, label='Desired Signal')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Amplification of a signal with Deep Learning-based DPD')
plt.legend()
plt.show()

############################################################################## Test every step ##############################################################################
frequency1 = 50       # Hz
duration = 1          # seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
signal2 = np.sin(2 * np.pi * frequency1 * t)
signalAfterInverse = inverseSaturationCurve(signal2)


plt.plot(t, signal2, label='Original Signal')
plt.plot(t, signalAfterInverse, label='Signal After Inverse Distortion')
plt.xlabel('Input Amplitude')
plt.ylabel('Output Amplitude')
plt.title('Effects of transformations on a test signal')
plt.legend()
plt.show()
