import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

initial_learningRate = 0.0001 # was 0.001
learningRateDecayRate = 0.955
dropout_rate = 0 # add in dropouts and try different values
numberOfEpochs = 100

print("Reading Data from Datasets...")
inputTrainData = np.genfromtxt('datasets/DPA_160MHz/train_input.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
outputTrainData = np.genfromtxt('datasets/DPA_160MHz/train_output.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

inputTestData = np.genfromtxt('datasets/DPA_160MHz/test_input.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
outputTestData = np.genfromtxt('datasets/DPA_160MHz/test_output.csv', delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
# inputTestData = inputTrainData[:1000, :]
# inputTrainData = inputTrainData[1001:, :]
# outputTestData = outputTrainData[:1000, :]
# outputTrainData = outputTrainData[1001:, :]

input_I_train = inputTrainData[:, 0]
input_Q_train = inputTrainData[:, 1]
output_I_train = outputTrainData[:, 0]
output_Q_train = outputTrainData[:, 1]
input_I_test = inputTestData[:, 0]
input_Q_test = inputTestData[:, 1]
output_I_test = outputTestData[:, 0]
output_Q_test = outputTestData[:, 1]

print("Data Read, now cleaning data")

# for i in range(len(outputTrainData)):
#     if not(outputTrainData[i][0] in output_I_train) or not(outputTrainData[i][1] in output_Q_train):
#         input_I_train.append(inputTrainData[i][0])
#         input_Q_train.append(inputTrainData[i][1])
#         output_I_train.append(outputTrainData[i][0])
#         output_Q_train.append(outputTrainData[i][1])
#     elif input_I_train.index(inputTrainData[i][0]) != input_Q_train.index(inputTrainData[i][1]):
#         input_I_train.append(inputTrainData[i][0])
#         input_Q_train.append(inputTrainData[i][1])
#         output_I_train.append(outputTrainData[i][0])
#         output_Q_train.append(outputTrainData[i][1])

# for i in range(len(outputTestData)):
#     if not(outputTestData[i][0] in output_I_test) or not(outputTestData[i][1] in output_Q_test):
#         input_I_test.append(inputTestData[i][0])
#         input_Q_test.append(inputTestData[i][1])
#         output_I_test.append(outputTestData[i][0])
#         output_Q_test.append(outputTestData[i][1])
#     elif input_I_test.index(inputTestData[i][0]) != input_Q_test.index(inputTestData[i][1]):
#         input_I_test.append(inputTestData[i][0])
#         input_Q_test.append(inputTestData[i][1])
#         output_I_test.append(outputTestData[i][0])
#         output_Q_test.append(outputTestData[i][1])

input_I_train = np.array(input_I_train)
input_Q_train = np.array(input_Q_train)
input_I_train = input_I_train.astype(np.float64)
input_Q_train = input_Q_train.astype(np.float64)
input_I_train = input_I_train.reshape(-1, 1)
input_Q_train = input_Q_train.reshape(-1, 1)
input_I_train = np.abs(input_I_train)
input_Q_train = np.abs(input_Q_train)

output_I_train = np.array(output_I_train)
output_Q_train = np.array(output_Q_train)
output_I_train = output_I_train.astype(np.float64)
output_Q_train = output_Q_train.astype(np.float64)
output_I_train = output_I_train.reshape(-1, 1)
output_Q_train = output_Q_train.reshape(-1, 1)
output_I_train = np.abs(output_I_train)
output_Q_train = np.abs(output_Q_train)

input_I_test = np.array(input_I_test)
input_Q_test = np.array(input_Q_test)
input_I_test = input_I_test.astype(np.float64)
input_Q_test = input_Q_test.astype(np.float64)
input_I_test = input_I_test.reshape(-1, 1)
input_Q_test = input_Q_test.reshape(-1, 1)

output_I_test = np.array(output_I_test)
output_Q_test = np.array(output_Q_test)
output_I_test = output_I_test.astype(np.float64)
output_Q_test = output_Q_test.astype(np.float64)
output_I_test = output_I_test.reshape(-1, 1)
output_Q_test = output_Q_test.reshape(-1, 1)

# --- 1. Define the Custom Dataset for Power Amplifier Data ---
class PowerAmplifierDataset(Dataset):
    """
    A custom PyTorch Dataset for power amplifier input-output data.
    It takes NumPy arrays for input (X) and output (Y).
    Now configured for 2 input features and 2 output features.
    """
    def __init__(self, inputs, outputs):
        """
        Initializes the dataset with input and output data.

        Args:
            inputs (np.ndarray): A NumPy array of input power values.
                                 Shape should be (num_samples, 2).
            outputs (np.ndarray): A NumPy array of corresponding output power values.
                                  Shape should be (num_samples, 2).
        """
        # Ensure inputs and outputs are 2D with shape (num_samples, num_features)
        if inputs.ndim == 1:
            raise ValueError("Inputs must be 2D. If you have 1D data, reshape to (num_samples, 1) or (num_samples, 2) as appropriate.")
        if outputs.ndim == 1:
            raise ValueError("Outputs must be 2D. If you have 1D data, reshape to (num_samples, 1) or (num_samples, 2) as appropriate.")

        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Inputs and outputs must have the same number of samples.")
        # --- CHANGED: Expecting 2 input and 2 output features ---
        if inputs.shape[1] != 2:
            raise ValueError(f"This model expects exactly 2 input features, but got {inputs.shape[1]}.")
        if outputs.shape[1] != 2:
            raise ValueError(f"This model expects exactly 2 output features, but got {outputs.shape[1]}.")

        # Convert NumPy arrays to PyTorch tensors with float32 dtype
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (input, output) pair by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (input_tensor, output_tensor) for the given index.
        """
        return self.inputs[idx], self.outputs[idx]

# --- 2. Define the Neural Network Model for the Power Amplifier ---
class PowerAmplifierModel(nn.Module):
    """
    A simple feed-forward neural network (MLP) to model a power amplifier.
    Now configured for 2 inputs and 2 outputs.
    """
    def __init__(self):
        """
        Initializes the layers of the neural network.
        We use a few hidden layers with ReLU activation, and a final linear
        layer for the regression output.
        """
        super(PowerAmplifierModel, self).__init__()
        # --- CHANGED: Input layer now takes 2 features ---
        self.fc1 = nn.Linear(2, 2)  # 2 inputs to first hidden layer
        self.relu1 = nn.LeakyReLU()       # Activation function

        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Hidden layer 2
        self.fc2 = nn.Linear(2, 128) # First hidden to second hidden layer
        self.relu2 = nn.LeakyReLU()       # Activation function

        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Hidden layer 3
        # self.fc3 = nn.Linear(8, 8) # Second hidden to third hidden layer
        # self.relu3 = nn.LeakyReLU()       # Activation function

        # self.dropout3 = nn.Dropout(p=dropout_rate)

        # self.fc4 = nn.Linear(8, 8)
        # self.relu4 = nn.ReLU()

        # self.fc5 = nn.Linear(32, 2)
        # self.relu5 = nn.ReLU()

        # --- CHANGED: Output layer now produces 2 features ---
        self.fc4 = nn.Linear(128, 2)   # Third hidden to output layer (2 outputs)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor (batch_size, 2).

        Returns:
            torch.Tensor: The output tensor (batch_size, 2).
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        # x = self.fc3(x)
        # x = self.relu3(x)
        # x = self.dropout3(x)
        # x = self.fc4(x)
        # x = self.relu4(x)
        # x = self.fc5(x)
        # x = self.relu5(x)
        x = self.fc4(x)
        return x

def MBELoss(predictions, targets):
    """
    Calculates the Mean Bias Error (MBE) loss.

    Args:
        predictions (torch.Tensor): The model's predicted values.
        targets (torch.Tensor): The true target values.

    Returns:
        torch.Tensor: The calculated MBE loss.
    """
    return torch.mean(predictions - targets)

def getLearningRate(epoch):
    return (initial_learningRate * (0.95 ** epoch))

# --- Main execution block (CRUCIAL for Windows multiprocessing with DataLoader) ---
if __name__ == '__main__':
    print("Starting Power Amplifier Model Training (2 Inputs, 2 Outputs)...")

    # --- 3. Generate Dummy Data (Replace with your actual NumPy arrays) ---
    num_samples = 1000
    np.random.seed(42) # for reproducibility

    # --- CHANGED: Generate 2 input features ---
    # Input feature 1 (e.g., input power 1)
    input_feature_1 = input_I_train
    # Input feature 2 (e.g., input power 2 or a control voltage)
    input_feature_2 = input_Q_train

    # Combine into a (num_samples, 2) array
    input_power_np_train = np.array((input_I_train, input_Q_train))
    input_power_np_test = np.array((output_I_test, output_Q_test))
    input_power_np_train = input_power_np_train.reshape(-1, 2) # Ensure shape is (num_samples, 2)
    input_power_np_test = input_power_np_test.reshape(-1, 2) # Ensure shape is (num_samples, 2)

    # --- CHANGED: Generate 2 output features based on the 2 inputs ---
    # Simulate non-linear amplifier responses for each output
    output_feature_1 = output_I_train
    output_feature_2 = output_Q_train

    # Combine into a (num_samples, 2) array
    output_power_np_train = np.array((output_I_train, output_Q_train))
    output_power_np_test = np.array((output_I_test, output_Q_test))
    output_power_np_train = output_power_np_train.reshape(-1, 2) # Ensure shape is (num_samples, 2)
    output_power_np_test = output_power_np_test.reshape(-1, 2) # Ensure shape is (num_samples, 2)


    print(f"Generated {num_samples} dummy data points.")
    print(f"Training Input power shape: {input_I_train.shape}")
    print(f"Training Output power shape: {output_I_train.shape}")
    print(f"Testing Input power shape: {input_I_test.shape}")
    print(f"Testing Output power shape: {input_Q_test.shape}")


    # --- 4. Create Dataset and DataLoader instances ---
    train_dataset = PowerAmplifierDataset(input_power_np_train, output_power_np_train)
    test_dataset = PowerAmplifierDataset(np.abs(input_power_np_test), np.abs(output_power_np_test))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"DataLoaders created with batch size: {batch_size}")

    # --- 5. Instantiate the Model, Loss Function, and Optimizer ---
    model = PowerAmplifierModel()
    criterion = nn.MSELoss()
    # def criterion(predictions, targets):
    #     return MBELoss(predictions, targets)
    # optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # Check for GPU availability and move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # --- 6. Training Loop ---
    num_epochs = numberOfEpochs
    train_losses = []
    test_losses = []
    learningRate = initial_learningRate

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        optimizer = optim.Adam(model.parameters(), lr=learningRate)
        model.train() # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs) # inputs here will have shape (batch_size, 2)
            loss = criterion(outputs, targets) # targets here will have shape (batch_size, 2)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Evaluation after each epoch ---
        model.eval() # Set the model to evaluation mode
        test_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations during evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0: # Print every 10 epochs or first epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}")
        learningRate = learningRate * learningRateDecayRate

    print("Training finished!")

    # --- 7. Visualize Training Progress (Loss over epochs) ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 8. Make Predictions and Visualize Results ---
    model.eval() # Set model to evaluation mode for final predictions
    with torch.no_grad():
        # Ensure X_test is (num_samples, 2) before converting to tensor
        # X_test_reshaped = X_test # Already 2D from train_test_split
        test_inputs_tensor = torch.tensor(np.abs(input_power_np_test), dtype=torch.float32).to(device)
        input_power_np_test_signs = np.heaviside(input_power_np_test, 1) - np.heaviside(-1*input_power_np_test, 1)
        predicted_outputs_tensor = model(test_inputs_tensor)

        # Move predictions back to CPU and convert to NumPy for plotting
        predicted_outputs_np = predicted_outputs_tensor.cpu().numpy()

 
    predicted_outputs_np = predicted_outputs_np.reshape(2, -1)
    input_power_np_test_signs = input_power_np_test_signs.reshape(2, -1)
    predicted_outputs_np = predicted_outputs_np * input_power_np_test_signs
    # Plot for Input 1 vs Output 1
    plt.figure(figsize=(12, 7))
    plt.scatter(predicted_outputs_np[0], predicted_outputs_np[1], color='red', linewidth=2, label='Predicted Output')
    plt.scatter(output_I_test, output_Q_test, s=10, label='Actual Output', alpha=0.6)
    plt.scatter(input_I_test, input_Q_test, s=10, label='Input', alpha=0.6)
    plt.xlabel('I value')
    plt.ylabel('Q value')
    plt.title('Power Amplifier Model: Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(predicted_outputs_np[0], predicted_outputs_np[1], color='red', linewidth=2, label='Predicted Output')
    plt.xlabel('I value')
    plt.ylabel('Q value')
    plt.title('Power Amplifier Model')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(output_I_test, output_Q_test, s=10, label='Actual Output', alpha=0.6)
    plt.xlabel('I value')
    plt.ylabel('Q value')
    plt.title('Power Amplifier Actual Output')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 9. Save the trained model ---
    model_path = 'rfpaModel.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --- Example of how to load the model ---
    # loaded_model = PowerAmplifierModel() # Re-instantiate the model architecture
    # loaded_model.load_state_dict(torch.load(model_path))
    # loaded_model.to(device)
    # loaded_model.eval() # Set to evaluation mode after loading
    # print("Model loaded successfully for inference.")

    # --- Example of making a single prediction with the loaded model ---
    # new_input_power = torch.tensor([[5.0, 10.0]], dtype=torch.float32).to(device) # Provide 2 input values
    # with torch.no_grad():
    #     predicted_output = loaded_model(new_input_power)
    # print(f"Prediction for inputs [5.0, 10.0]: {predicted_output.cpu().numpy()}")
