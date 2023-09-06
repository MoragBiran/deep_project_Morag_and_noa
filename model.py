import torch.nn as nn
import torch.nn.init as init

# In the process of creating a neural network, 22 features are entered, passed through
# layers of 256, 128, 64, and finally, a binary output is produced
class FCNet(nn.Module):
    def __init__(self, input_size=22, num_classes=1):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        #Disables a weight or a neuron randomly to make the model more challenging.
        # It makes it more difficult for the trainee to study harder so that the test will be easier for him.
        self.dropout = nn.Dropout(0.5)
        
    #Concatenation of the neural network
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
