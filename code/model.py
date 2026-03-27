import torch
import torch.nn as nn

class VC_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VC_GNN, self).__init__()
        self.gnn_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (batch, frames, channels, h, w)
        x = self.relu(self.gnn_layer(x))
        x = self.fc(x)
        return x
