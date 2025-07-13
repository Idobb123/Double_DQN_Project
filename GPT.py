import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, num_actions: int):
        super(DeepQNetwork, self).__init__()
        
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
        with torch.no_grad():
            input = torch.zeros(1, *shape)  # Batch size 1
            x = F.relu(self.conv1(input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.out(x)

# Example usage:
if __name__ == "__main__":
    num_actions = 6  # Example number of valid actions
    model = DeepQNetwork(num_actions)
    dummy_input = torch.zeros(1, 4, 84, 84)  # batch size 1, 4 stacked grayscale frames
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [1, num_actions]
