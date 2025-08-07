####################################################################################### Imports #######################################################################################
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rfpaModel import PowerAmplifierModel
from sklearn.model_selection import train_test_split

####################################################################################### Constants #######################################################################################
threshold = 1.2
highValue = 3
degrees = 20
sampling_rate = 200000000 # in Hz

############################################################################# Set up the inverse RFPA Model #############################################################################
inverse_model_path = 'inverse_pa_model.pth'

# Check for GPU availability and move model to GPU if available
# inverse_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inverse_device = torch.device("cpu")


inverse_loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
inverse_loaded_model.to(inverse_device)
print(f"Using device: {inverse_device}")
inverse_loaded_model.load_state_dict(torch.load(inverse_model_path))
inverse_loaded_model.to(inverse_device)
inverse_loaded_model.eval() # Set to evaluation mode after loading
print("Inverse Model loaded successfully for inference.")

#################################################################### Define the inverse saturation curve ####################################################################

def inverseSaturationCurve(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(inverse_device)
        with torch.no_grad():
            predicted_output.append(inverse_loaded_model(new_input_power))
    # new_output = []
    # for i in range(len(predicted_output)):
    #     new_output.append(predicted_output[i].cpu().numpy())
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

######################################################################## Define the rotation ########################################################################
t = np.linspace(0, 1000, 1000, endpoint=False)
sampling_rate = 1000  # Hz
frequency1 = 50       # Hz
duration = 1          # seconds
amplitude1 = 0.5
x0 = amplitude1 * np.sin(2 * np.pi * frequency1 * t) + 0.7
x1 = inverseSaturationCurve(x0)
theta = np.deg2rad(degrees)
rotation = [[np.cos(theta), -1*np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]]
def rotate(xArr, yArr):
    return np.dot(rotation, [xArr, yArr])
inputData = np.array([x0, x1])
rotatedData = np.dot(rotation, inputData)

################################################################################## Machine Learning ##################################################################################

# 1. Define the 5-layer MLP Neural Network
# The network will have 2 inputs, hidden layers of 64, 128, and 64 neurons,
# and 2 outputs.
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Input layer (2 features) to first hidden layer (64 neurons)
        self.fc1 = nn.Linear(2, 64)
        # First hidden layer (64 neurons) to second hidden layer (128 neurons)
        self.fc2 = nn.Linear(64, 128)
        # Second hidden layer (128 neurons) to third hidden layer (64 neurons)
        self.fc3 = nn.Linear(128, 64)
        # Third hidden layer (64 neurons) to output layer (2 features)
        self.fc4 = nn.Linear(64, 2)

        # Activation function (ReLU is commonly used for hidden layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through the first fully connected layer and apply ReLU
        x = self.relu(self.fc1(x))
        # Pass through the second fully connected layer and apply ReLU
        x = self.relu(self.fc2(x))
        # Pass through the third fully connected layer and apply ReLU
        x = self.relu(self.fc3(x))
        # Pass through the final fully connected layer (output layer)
        # No activation function here for regression tasks, or use
        # softmax for classification if outputs are probabilities.
        # For 2 outputs as specified, assuming regression or direct scores.
        x = self.fc4(x)
        return x

# 2. Prepare your training data (using NumPy matrices as requested)
# Let's create some dummy data for demonstration.
# In a real scenario, you would load your actual data here.

# Input data (features) - a NumPy array with shape (num_samples, 2)
# For example, 100 samples, each with 2 input features.
inputData = inputData.reshape(-1, 2)
rotatedData = rotatedData.reshape(-1, 2)
temp = []
for i in range(800):
    temp.append(inputData[i])
X_train_np = np.array(temp)
temp = []
for i in range(800):
    temp.append(rotatedData[i])
y_train_np = np.array(temp)

X_train_np = X_train_np.reshape(-1, 2)
y_train_np = y_train_np.reshape(-1, 2)

# print(f"Shape of X_train_np: {X_train_np.shape}")
# print(f"Shape of y_train_np: {y_train_np.shape}")
# print(f"Sample X_train_np:\n{X_train_np[:5]}")
# print(f"Sample y_train_np:\n{y_train_np[:5]}")

# 3. Convert NumPy arrays to PyTorch Tensors
# PyTorch models operate on Tensors.
X_train_tensor = torch.from_numpy(X_train_np).float()
y_train_tensor = torch.from_numpy(y_train_np).float()

# print(f"\nShape of X_train_tensor: {X_train_tensor.shape}")
# print(f"Shape of y_train_tensor: {y_train_tensor.shape}")
# print(f"Type of X_train_tensor: {X_train_tensor.dtype}")
# print(f"Type of y_train_tensor: {y_train_tensor.dtype}")

# 4. Instantiate the model, define loss function and optimizer
model = MLP()
model.float()

# For regression tasks, Mean Squared Error (MSE) is a common loss function.
# For classification, CrossEntropyLoss is typical.
criterion = nn.MSELoss()

# Adam optimizer is a good general-purpose optimizer.
# You can adjust the learning rate (lr).
optimizer = optim.Adam(model.parameters(), lr=0.001)

# print(f"\nModel architecture:\n{model}")

# 5. Training Loop
num_epochs = 1000 # Number of times to iterate over the entire dataset

print("\nStarting training...")
if __name__ == '__main__':
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # 5.1. Forward pass: Compute predicted y by passing x to the model
        outputs = model(X_train_tensor)

        # 5.2. Compute loss
        loss = criterion(outputs, y_train_tensor)

        # 5.3. Backward pass: Zero gradients, compute gradients, update weights
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients of loss with respect to model parameters
        optimizer.step()      # Update model parameters using the gradients

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    # 6. Make predictions with the trained model
    # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    model.float()

    temp = []
    for i in range(200):
        temp.append(inputData[800+i])
    X_new_np = np.array(temp)
    temp = []
    for i in range(200):
        temp.append(rotatedData[800+i])
    y_new_np = np.array(temp)

    X_new_np = X_new_np.reshape(-1, 2)
    y_new_np = y_new_np.reshape(-1, 2)

    X_new_tensor = torch.from_numpy(X_new_np).float()

    # Predict
    with torch.no_grad(): # Disable gradient calculation for inference
        predictions = model(X_new_tensor)

    # print(f"\nNew input data (NumPy):\n{X_new_np}")
    # print(f"Predicted outputs (Tensor):\n{predictions}")
    # print(f"Predicted outputs (NumPy):\n{predictions.numpy()}")

    # Compare with expected outputs based on our dummy data relationship
    X_new_np = X_new_np.reshape(2, -1)
    y_new_np = y_new_np.reshape(2, -1)
    predictions = predictions.reshape(2, -1)
    input_x_values = X_new_np[0]
    input_y_values = X_new_np[1]
    actualOutput_x_values = y_new_np[0]
    actualOutput_y_values = y_new_np[1]
    predictedOutput_x_values = predictions[0]
    predictedOutput_y_values = predictions[1]

    print(f"\nShape of input_x_values: {input_x_values.shape}")
    print(f"Shape of input_y_values: {input_y_values.shape}")
    print(f"\nShape of actualOutput_x_values: {actualOutput_x_values.shape}")
    print(f"Shape of actualOutput_y_values: {actualOutput_y_values.shape}")
    print(f"\nShape of predictedOutput_x_values: {predictedOutput_x_values.shape}")
    print(f"Shape of predictedOutput_y_values: {predictedOutput_y_values.shape}")

    plt.scatter(input_x_values, input_y_values, label='Input Data')
    plt.scatter(actualOutput_x_values, actualOutput_y_values, label='Actual Output Data')
    plt.plot(predictedOutput_x_values, predictedOutput_y_values, label='Predicted Output Data')
    plt.legend()
    plt.show()


    # --- 9. Save the trained model ---
    # Recommended way to save: save the state_dict (model parameters)
    model_path = 'rotated_inverse_pa_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
