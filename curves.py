import numpy as np
import matplotlib.pyplot as plt
from rfpaModel import PowerAmplifierModel, PowerAmplifierDataset
import torch

paGain = 3.2


forward_model_path = 'rfpaModel.pth'
# Check for GPU availability and move model to GPU if available
forward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {forward_device}")
forward_model = PowerAmplifierModel() # Re-instantiate the model architecture
forward_model.load_state_dict(torch.load(forward_model_path))
forward_model.to(forward_device)
forward_model.eval() # Set to evaluation mode after loading
print("RFPA Model loaded successfully for inference.")

inverse_model_path = 'inverseRFPAModel.pth'
# Check for GPU availability and move model to GPU if available
inverse_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {inverse_device}")
inverse_model = PowerAmplifierModel() # Re-instantiate the model architecture
inverse_model.load_state_dict(torch.load(inverse_model_path))
inverse_model.to(inverse_device)
inverse_model.eval() # Set to evaluation mode after loading
print("Inverse RFPA Model loaded successfully for inference.")

def RFPA(I, Q):
    new_input_power = torch.tensor([[I, Q]], dtype=torch.float32).to(forward_device) # Provide 2 input values
    with torch.no_grad():
        predicted_output = forward_model(new_input_power.reshape(-1, 2))
        predicted_output = predicted_output.cpu().numpy()
    return predicted_output.reshape(2, -1)

def inverseRFPA(I, Q):
    new_input_power = torch.tensor([[I, Q]], dtype=torch.float32).to(inverse_device) # Provide 2 input values
    with torch.no_grad():
        predicted_output = inverse_model(new_input_power.reshape(-1, 2))
        predicted_output = predicted_output.cpu().numpy()
    return predicted_output.reshape(2, -1)

theta = np.linspace(0, 2*np.pi, 1000)
mag = np.linspace(0, 3, 1000)
input_I = np.cos(theta) * mag
input_Q = np.sin(theta) * mag
forwardModel_output = RFPA(input_I, input_Q)
inverseModel_output = inverseRFPA(input_I, input_Q)
cascaded_output = RFPA(inverseRFPA(input_I * paGain, input_Q * paGain)[0], inverseRFPA(input_I * paGain, input_Q * paGain)[1])
desired_I = input_I * paGain
desired_Q = input_Q * paGain

plt.figure(figsize=(7, 7))
plt.scatter(forwardModel_output[0], forwardModel_output[1], s=10, label='Output Without DPD', alpha=0.6)
plt.scatter(cascaded_output[0], cascaded_output[1], s=10, label='Output With DPD', alpha=0.6)
plt.scatter(desired_I, desired_Q, s=10, label='Desired Output', alpha = 0.6)
plt.scatter(input_I, input_Q, s=10, label='Input Signal', alpha=0.6)
plt.xlabel('I value')
plt.ylabel('Q value')
plt.title('Model Predictions')
plt.legend()
plt.show()


input_power_mag = np.sqrt(input_I**2 + input_Q**2)
forwardModel_output_mag = np.sqrt(forwardModel_output[0]**2 + forwardModel_output[1]**2)
inverseModel_output_mag = np.sqrt(inverseModel_output[0]**2 + inverseModel_output[1]**2)
cascaded_output_mag = np.sqrt(cascaded_output[0]**2 + cascaded_output[1]**2)
desired_output_mag = np.sqrt(desired_I**2 + desired_Q**2)
plt.plot(input_power_mag, forwardModel_output_mag, label='RFPA Model')
plt.plot(input_power_mag, inverseModel_output_mag, label='Inverse RFPA Model')
plt.plot(input_power_mag, cascaded_output_mag, label='Cascaded Output')
plt.plot(input_power_mag, desired_output_mag, label='Desired Output')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('RFPA Model Response')
plt.legend()
plt.show()


t = np.linspace(0, 1000, 1000)
inputSignal = 0.2 * np.sin(0.1 * t) + 0.08 * np.sin(0.08 * t) + 0.2 * (np.random.randn(len(t)) / 10.0) + 0.5

desiredSignal = inputSignal * paGain

