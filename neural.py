import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 128], output_dim=2, with_softmax=True):
        super(MLP, self).__init__()
        self.with_softmax = with_softmax

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(0.3) # Added dropout
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(0.3) # Added dropout
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        if with_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.fc1(x)))  # first hidden layer
        x = self.dropout2(F.relu(self.fc2(x)) ) # second hidden layer
        x = self.fc3(x)          # output layer (no activation here)
        if self.with_softmax:
            x = self.softmax(x)
        return x
    
class OptimizedMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[512, 128], output_dim=2, with_softmax=True):
        super(OptimizedMLP, self).__init__()
        self.with_softmax = with_softmax

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.norm1 = nn.LayerNorm(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.norm2 = nn.LayerNorm(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        # Optional softmax (use nn.LogSoftmax + NLLLoss instead for numerical stability)
        if with_softmax:
            self.softmax = nn.Softmax(dim=1)

        # Optional: Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x