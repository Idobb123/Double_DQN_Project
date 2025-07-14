import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CUDA device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_actions: int):
        super(DQN, self).__init__()
        
        # First conv layer: 32 filters of 8x8, stride 4
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        
        # Second conv layer: 64 filters of 4x4, stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Third conv layer: 64 filters of 3x3, stride 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Compute the flattened conv output size dynamically
        self._conv_output_size = self._get_conv_output((4, 84, 84))
        
        # Fully connected layer with 512 rectifier units
        self.fc1 = nn.Linear(self._conv_output_size, 512)
        
        # Output layer: one output per valid action
        self.out = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        bs = 1
        # Create a dummy input on the same device as the model
        input = torch.zeros(bs, *shape).to(next(self.parameters()).device)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))


    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.out(x)
    
def state_to_dqn_input(state_np):
    """
    Convert a (84, 84, 4) numpy array into a torch tensor for DQN.
    """
    # Convert to float tensor and normalize if needed
    tensor = torch.tensor(state_np, dtype=torch.float32, device=device)  # shape: (84, 84, 4)

    # Add batch dimension and permute to channel-first: (1, 4, 84, 84)
    tensor = tensor.unsqueeze(0)

    return tensor

# Example usage:
if __name__ == "__main__":
    num_actions = 6  # Example number of valid actions
    model = DQN(num_actions).to(device)
    dummy_input = torch.zeros(1, 4, 84, 84).to(device)  # batch size 1, 4 stacked grayscale frames
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [1, num_actions]
