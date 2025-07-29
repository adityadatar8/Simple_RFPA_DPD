import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rfpaModel import PowerAmplifierModel, PowerAmplifierDataset

forward_model_path = 'rfpaModel.pth'
# Check for GPU availability and move model to GPU if available
forward_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {forward_device}")
forward_model = PowerAmplifierModel() # Re-instantiate the model architecture
forward_model.load_state_dict(torch.load(forward_model_path))
forward_model.to(forward_device)
forward_model.eval() # Set to evaluation mode after loading
print("RFPA Model loaded successfully for inference.")

def RFPA(I, Q):
    new_input_power = torch.tensor([[I, Q]], dtype=torch.float32).to(forward_device) # Provide 2 input values
    with torch.no_grad():
        predicted_output = forward_model(new_input_power.reshape(-1, 2))
        predicted_output = predicted_output.cpu().numpy()
    return predicted_output

num_samples_train = 23040
theta_train = np.linspace(0, 2*np.pi, num_samples_train)
mag_train = np.linspace(0, 5, num_samples_train)
forwardInput_I_train = np.cos(theta_train) * mag_train
forwardInput_Q_train = np.sin(theta_train) * mag_train
forwardOutput_train = RFPA(forwardInput_I_train, forwardInput_Q_train).reshape(2, -1)
inverseInput_I_train = forwardOutput_train[0]
inverseInput_Q_train = forwardOutput_train[1]
inverseOutput_I_train = forwardInput_I_train
inverseOutput_Q_train = forwardInput_Q_train

num_samples_test = 1000
theta_test = np.linspace(0, 2*np.pi, num_samples_test)
mag_test = np.linspace(0, 5, num_samples_test)
forwardInput_I_test = np.cos(theta_test) * mag_test
forwardInput_Q_test = np.sin(theta_test) * mag_test
forwardOutput_test = RFPA(forwardInput_I_test, forwardInput_Q_test).reshape(2, -1)
inverseInput_I_test = forwardOutput_test[0]
inverseInput_Q_test = forwardOutput_test[1]
inverseOutput_I_test = forwardInput_I_test
inverseOutput_Q_test = forwardInput_Q_test

# --- 1. Define the Custom Dataset for Power Amplifier Data ---
class InversePowerAmplifierDataset(Dataset):
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
class InversePowerAmplifierModel(nn.Module):
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
        super(InversePowerAmplifierModel, self).__init__()
        # --- CHANGED: Input layer now takes 2 features ---
        self.fc1 = nn.Linear(2, 64)  # 2 inputs to first hidden layer
        self.relu1 = nn.ReLU()       # Activation function

        # Hidden layer 2
        self.fc2 = nn.Linear(64, 128) # First hidden to second hidden layer
        self.relu2 = nn.ReLU()       # Activation function

        # Hidden layer 3
        self.fc3 = nn.Linear(128, 64) # Second hidden to third hidden layer
        self.relu3 = nn.ReLU()       # Activation function

        # --- CHANGED: Output layer now produces 2 features ---
        self.fc4 = nn.Linear(64, 2)   # Third hidden to output layer (2 outputs)

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
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x) # No activation here, as it's a regression task
        return x

# --- Main execution block (CRUCIAL for Windows multiprocessing with DataLoader) ---
if __name__ == '__main__':
    print("Starting Power Amplifier Model Training (2 Inputs, 2 Outputs)...")

    # --- 3. Generate Dummy Data (Replace with your actual NumPy arrays) ---
    num_samples = 1000
    np.random.seed(42) # for reproducibility

    # --- CHANGED: Generate 2 input features ---
    # Input feature 1 (e.g., input power 1)
    input_feature_1 = inverseInput_I_train
    # Input feature 2 (e.g., input power 2 or a control voltage)
    input_feature_2 = inverseInput_Q_train

    # Combine into a (num_samples, 2) array
    input_power_np_train = np.array((inverseInput_I_train, inverseInput_Q_train))
    input_power_np_test = np.array((inverseOutput_I_test, inverseOutput_Q_test))
    input_power_np_train = input_power_np_train.reshape(-1, 2) # Ensure shape is (num_samples, 2)
    input_power_np_test = input_power_np_test.reshape(-1, 2) # Ensure shape is (num_samples, 2)

    # --- CHANGED: Generate 2 output features based on the 2 inputs ---
    # Simulate non-linear amplifier responses for each output
    output_feature_1 = inverseOutput_I_train
    output_feature_2 = inverseOutput_Q_train

    # Combine into a (num_samples, 2) array
    output_power_np_train = np.array((inverseOutput_I_train, inverseOutput_Q_train))
    output_power_np_test = np.array((inverseOutput_I_test, inverseOutput_Q_test))
    output_power_np_train = output_power_np_train.reshape(-1, 2) # Ensure shape is (num_samples, 2)
    output_power_np_test = output_power_np_test.reshape(-1, 2) # Ensure shape is (num_samples, 2)


    print(f"Generated {num_samples} dummy data points.")
    print(f"Training Input power shape: {input_power_np_train.shape}")
    print(f"Training Output power shape: {output_power_np_train.shape}")
    print(f"Testing Input power shape: {input_power_np_test.shape}")
    print(f"Testing Output power shape: {output_power_np_test.shape}")


    # --- 4. Create Dataset and DataLoader instances ---
    train_dataset = InversePowerAmplifierDataset(input_power_np_train, output_power_np_train)
    test_dataset = InversePowerAmplifierDataset(input_power_np_test, output_power_np_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"DataLoaders created with batch size: {batch_size}")

    # --- 5. Instantiate the Model, Loss Function, and Optimizer ---
    model = InversePowerAmplifierModel()
    criterion = nn.MSELoss() # Mean Squared Error is common for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam is a good general-purpose optimizer

    # Check for GPU availability and move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # --- 6. Training Loop ---
    num_epochs = 100
    train_losses = []
    test_losses = []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
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
        test_inputs_tensor = torch.tensor(input_power_np_test, dtype=torch.float32).to(device)
        predicted_outputs_tensor = model(test_inputs_tensor)

        # Move predictions back to CPU and convert to NumPy for plotting
        predicted_outputs_np = predicted_outputs_tensor.cpu().numpy()

    predicted_outputs_np = predicted_outputs_np.reshape(2, -1)
    # Plot for Input 1 vs Output 1
    plt.figure(figsize=(12, 7))
    plt.scatter(inverseOutput_I_test, inverseOutput_Q_test, s=10, label='Actual Output', alpha=0.6)
    plt.scatter(inverseInput_I_test, inverseInput_Q_test, s=10, label='Input', alpha=0.6)
    plt.plot(predicted_outputs_np[0], predicted_outputs_np[1], color='red', linewidth=2, label='Predicted Output')
    plt.xlabel('I value')
    plt.ylabel('Q value')
    plt.title('Inverse Power Amplifier Model: Actual vs. Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()


    # --- 9. Save the trained model ---
    model_path = 'InverseRFPAModel.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
