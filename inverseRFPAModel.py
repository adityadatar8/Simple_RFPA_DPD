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

################################################################################ Define the saturation curve ################################################################################
x = np.linspace(0, 1, 23040)

def saturationCurve(val):
    predicted_output = []
    for i in range(len(val)):
        new_input_power = torch.tensor([[val[i]]], dtype=torch.float32).to(device)
        with torch.no_grad():
            predicted_output.append(loaded_model(new_input_power))
    predicted_output = np.array(predicted_output)
    predicted_output = predicted_output.reshape(-1)
    return predicted_output

f = saturationCurve(x)


################################################################################## Machine Learning ##################################################################################

# --- 1. Define the Custom Dataset for Power Amplifier Data ---
class PowerAmplifierDataset(Dataset):
    """
    A custom PyTorch Dataset for power amplifier input-output data.
    It takes NumPy arrays for input (X) and output (Y).
    """
    def __init__(self, inputs, outputs):
        """
        Initializes the dataset with input and output data.

        Args:
            inputs (np.ndarray): A NumPy array of input power values.
                                 Shape should be (num_samples, 1).
            outputs (np.ndarray): A NumPy array of corresponding output power values.
                                  Shape should be (num_samples, 1).
        """
        if inputs.shape[0] != outputs.shape[0]:
            raise ValueError("Inputs and outputs must have the same number of samples.")
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1) # Ensure it's (num_samples, 1)
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1) # Ensure it's (num_samples, 1)

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
    It takes one input (e.g., input power) and produces one output (e.g., output power).
    """
    def __init__(self):
        """
        Initializes the layers of the neural network.
        We use a few hidden layers with ReLU activation, and a final linear
        layer for the regression output.
        """
        super(PowerAmplifierModel, self).__init__()
        # Input layer: 1 input feature
        self.fc1 = nn.Linear(1, 64)  # Input to first hidden layer
        self.relu1 = nn.ReLU()       # Activation function

        # Hidden layer 2
        self.fc2 = nn.Linear(64, 128) # First hidden to second hidden layer
        self.relu2 = nn.ReLU()       # Activation function

        # Hidden layer 3
        self.fc3 = nn.Linear(128, 64) # Second hidden to third hidden layer
        self.relu3 = nn.ReLU()       # Activation function

        # Output layer: 1 output feature (for regression)
        self.fc4 = nn.Linear(64, 1)   # Third hidden to output layer

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor (batch_size, 1).

        Returns:
            torch.Tensor: The output tensor (batch_size, 1).
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x) # No activation here, as it's a regression task (predicting continuous values)
        return x

# --- Main execution block (CRUCIAL for Windows multiprocessing with DataLoader) ---
if __name__ == '__main__':
    print("Starting Power Amplifier Model Training...")

    # --- 3. Generate Dummy Data (Replace with your actual NumPy arrays) ---
    # For demonstration, let's create a non-linear relationship like y = x^2 + 0.5x + noise
    num_samples = 1000
    np.random.seed(42) # for reproducibility

    # # Input power (e.g., in dBm or Watts) - let's say from 0 to 10
    # input_power_np = np.linspace(0, 10, num_samples).reshape(-1, 1)
    # # Simulate a non-linear amplifier response with some noise
    # output_power_np = (input_power_np**2 * 0.1 + input_power_np * 0.5 + np.random.randn(num_samples, 1) * 0.5)

    print(f"Generated {num_samples} dummy data points.")
    print(f"Input power shape: {f.shape}")
    print(f"Output power shape: {x.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        f, x, test_size=0.2, random_state=42
    )
    
    X_test = X_test.reshape(-1, 1) # Ensure it's (num_samples, 1)
    y_test = y_test.reshape(-1, 1) # Ensure it's (num_samples, 1)

    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    # --- 4. Create Dataset and DataLoader instances ---
    train_dataset = PowerAmplifierDataset(X_train, y_train)
    test_dataset = PowerAmplifierDataset(X_test, y_test)

    batch_size = 32
    # num_workers > 0 is recommended for faster data loading on multi-core CPUs,
    # but requires the `if __name__ == '__main__':` guard on Windows.
    # Set to 0 if you encounter multiprocessing issues that the guard doesn't fix.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"DataLoaders created with batch size: {batch_size}")

    # --- 5. Instantiate the Model, Loss Function, and Optimizer ---
    model = PowerAmplifierModel()
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
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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
        # Predict on the entire test set
        test_inputs_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predicted_outputs_tensor = model(test_inputs_tensor)

        # Move predictions back to CPU and convert to NumPy for plotting
        predicted_outputs_np = predicted_outputs_tensor.cpu().numpy()

    # Sort test inputs for a cleaner plot if the original data was sorted
    sort_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sort_indices]
    y_test_sorted = y_test[sort_indices]
    predicted_outputs_sorted = predicted_outputs_np[sort_indices]

    plt.figure(figsize=(12, 7))
    plt.scatter(X_test_sorted, y_test_sorted, s=10, label='Actual Test Data', alpha=0.6)
    plt.plot(X_test_sorted, predicted_outputs_sorted, color='red', linewidth=2, label='Model Predictions')
    plt.xlabel('Input Power')
    plt.ylabel('Output Power')
    plt.title('Inverse Power Amplifier Model: Actual vs. Predicted Output')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 9. Save the trained model ---
    # Recommended way to save: save the state_dict (model parameters)
    model_path = 'inverse_pa_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
