import numpy as np
import matplotlib.pyplot as plt
from rfpaModel import PowerAmplifierModel, PowerAmplifierDataset
import torch


model_path = 'rfpaModel.pth'
# Check for GPU availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.to(device)
loaded_model.eval() # Set to evaluation mode after loading
print("RFPA Model loaded successfully for inference.")

def RFPA(I, Q):
    new_input_power = torch.tensor([[I, Q]], dtype=torch.float32).to(device) # Provide 2 input values
    with torch.no_grad():
        predicted_output = loaded_model(new_input_power.reshape(-1, 2))
        predicted_output = predicted_output.cpu().numpy()
    return predicted_output

theta = np.linspace(0, 2*np.pi, 1000)
mag = np.linspace(0, 3, 1000)
input_I = np.cos(theta) * mag
input_Q = np.sin(theta) * mag
predicted_output = RFPA(input_I, input_Q).reshape(2, -1)

plt.figure(figsize=(7, 7))
plt.scatter(predicted_output[0], predicted_output[1], s=10, label='Predicted Output', alpha=0.6)
plt.scatter(input_I, input_Q, s=10, label='Input', alpha=0.6)
plt.xlabel('I value')
plt.ylabel('Q value')
plt.title('Model Predictions')
plt.legend()
plt.show()


input_power_mag = np.sqrt(input_I**2 + input_Q**2)
predicted_output_mag = np.sqrt(predicted_output[0]**2 + predicted_output[1]**2)
plt.plot(input_power_mag, predicted_output_mag, label='RFPA Model Output Power')
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('RFPA Model Response')
plt.legend()
plt.show()
