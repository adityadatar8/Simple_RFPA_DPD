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
threshold = 1
highValue = 3
degrees = 20
sampling_rate = 200000000 # in Hz
paGain = 3.1

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

############################################################################# Set up the Rotation Model #############################################################################
rotation_model_path = 'rotated_inverse_pa_model.pth'

# Check for GPU availability and move model to GPU if available
rotation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


rotation_loaded_model = MLP() # Re-instantiate the model architecture
rotation_loaded_model.float()
rotation_loaded_model.to(rotation_device)
print(f"Using device: {rotation_device}")
rotation_loaded_model.load_state_dict(torch.load(rotation_model_path))
rotation_loaded_model.to(rotation_device)
rotation_loaded_model.eval() # Set to evaluation mode after loading
print("Rotation Model loaded successfully for inference.")

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

# def rotateML(val):
#     predicted_output = []
#     for i in range(len(val)):
#         new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(rotation_device)
#         with torch.no_grad():
#             predicted_output.append(rotation_loaded_model(new_input_power))
#     predicted_output = np.array(predicted_output)
#     predicted_output = predicted_output.reshape(2, -1)
#     return predicted_output

def rotateML(val):
    # Convert to PyTorch Tensor and ensure float32
    inference_data_tensor = torch.from_numpy(val).float()

    # Make predictions
    with torch.no_grad():
        predictions_loaded_model = rotation_loaded_model(inference_data_tensor)
        return predictions_loaded_model.numpy()

######################################################################## Define the FIR filter ########################################################################

# Possibly need to filter out the end of the inverse saturation curve
def filter(val):
    return (val*np.heaviside(-1*(val-threshold), 1) + highValue*np.heaviside(val-threshold, 1))

linearizedOut = saturationCurve(rotatedInverseSaturation[1])
linearizedIn = filter(rotatedInverseSaturation[0])

############################################################################## Test every step ##############################################################################
sampling_rate = 1000  # Hz
frequency1 = 50       # Hz
duration = 1          # seconds
amplitude1 = 0.2
amplitude2 = 0.08
t = np.linspace(0, 1000, 1000)
# signal2 = amplitude1 * np.sin(0.1 * t) + 0.7
signal2 = amplitude1 * np.sin(0.1 * t) + amplitude2 * np.sin(0.08 * t) + 0.2 * (np.random.randn(len(t)) / 10.0) + 0.5 # Add some noise and DC offset
signal2 = filter(signal2)
signalAfterInverse = inverseSaturationCurve(signal2)
rotationInputMatrix = np.array([signal2, signalAfterInverse])

# Reshape rotationInputMatrix (convert 2x1000 into 1000x2)
result = []
for i in range(len(rotationInputMatrix[0])):
    result.append(np.array([rotationInputMatrix[0][i], rotationInputMatrix[1][i]]))
rotationInputMatrix = np.array(result)

signalAfterRotation = rotateML(rotationInputMatrix)

# Reshape signalAfterRotation (convert 1000x2 into 2x1000)
result = [[], []]
for i in range(len(signalAfterRotation)):
    result[0].append(signalAfterRotation[i][0])
    result[1].append(signalAfterRotation[i][1])
result[0] = np.array(result[0])
result[1] = np.array(result[1])
signalAfterRotation = np.array(result)

cascadedSignalRotation = saturationCurve(signalAfterRotation[1])
signalAfterMultiplication = signalAfterInverse * paGain
cascadedSignalMultiplier = saturationCurve(signalAfterMultiplication)
desiredSignal = signal2 * paGain


plt.plot(t, signal2, label='Original Signal')
plt.plot(t, signalAfterInverse, label='Signal After Inverse Distortion')
plt.plot(t, signalAfterMultiplication, label='Signal After Multiplication')
plt.plot(t, cascadedSignalRotation, label='Cascaded Signal with Rotation')
plt.plot(t, signalAfterRotation[1], label='Signal After Rotation')
plt.plot(t, desiredSignal, label='Desired Signal')
plt.plot(t, cascadedSignalMultiplier, label='Cascaded Signal With Constant Multiplier')
plt.xlabel('Time')
plt.ylabel('Output Amplitude')
plt.title('Effects of transformations on a test signal')
plt.legend()
plt.show()

x = np.linspace(0, 2, 1000)
f = saturationCurve(x)
g = inverseSaturationCurve(x)
plt.plot(x, f, label='PA Model')
plt.plot(x, g, label='Inverse PA Model')

# Reshape rotationInputMatrix (convert 2x1000 into 1000x2)
rotationInputMatrix = np.array([x, g])
result = []
for i in range(len(rotationInputMatrix[0])):
    result.append(np.array([rotationInputMatrix[0][i], rotationInputMatrix[1][i]]))
rotationInputMatrix = np.array(result)

signalAfterRotation = rotateML(rotationInputMatrix)

# Reshape signalAfterRotation (convert 1000x2 into 2x1000)
result = [[], []]
for i in range(len(signalAfterRotation)):
    result[0].append(signalAfterRotation[i][0])
    result[1].append(signalAfterRotation[i][1])
result[0] = np.array(result[0])
result[1] = np.array(result[1])
signalAfterRotation = np.array(result)

plt.plot(x, signalAfterRotation[1], label='Rotated Curve')
plt.plot(x, saturationCurve(signalAfterRotation[1]), label='Cascaded Curve')
plt.plot(filter(x), saturationCurve(g * paGain), label='Constant Multiplier after Inverse')
plt.xlabel('Input Amplitude')
plt.ylabel('Output Amplitude')
plt.title('Output Power vs. Input Power with different transformations')
plt.legend()
plt.show()

regularAmplifiedSignal = saturationCurve(signal2)
mse = np.mean((cascadedSignalMultiplier - desiredSignal)**2)
mseRegular = np.mean((regularAmplifiedSignal - desiredSignal)**2)
print("MSE with DPD: " + str(mse))
print("MSE without DPD: " + str(mseRegular))

plt.plot(t, signal2, label='Original Signal')
plt.plot(t, cascadedSignalMultiplier, label='Cascaded Signal with Constant Multiplier')
plt.plot(t, regularAmplifiedSignal, label='Output of PA without predistortion')
plt.plot(t, desiredSignal, label='Desired Signal')
plt.xlabel('Time')
plt.ylabel('Output Amplitude')
plt.title('Comparing amplification with DPD and without DPD')
plt.legend()
plt.show()

