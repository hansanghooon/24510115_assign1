import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer with a dropout probability of 0.5
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.5)  # Another Dropout layer with the same probability
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Convolution and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first fully connected layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second fully connected layer
        x = self.fc3(x)
        
        return x



class CustomMLP(nn.Module):


    def __init__(self):
        super(CustomMLP, self).__init__()
        # Convolution layers
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout layer with a dropout probability of 0.3
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.2)  # Another Dropout layer with the same probability
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Convolution and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first fully connected layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second fully connected layer
        x = self.fc3(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = LeNet5()
num_params = count_parameters(model)
print(f"LeNet-5 모델의 파라메터 수: {num_params}")
model = CustomMLP()
num_params = count_parameters(model)
print(f"CustomMLP 모델의 파라메터 수: {num_params}")