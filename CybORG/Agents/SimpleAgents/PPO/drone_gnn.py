import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
class DroneGNN(nn.Module):
    """Graph Neural Network for processing drone swarm netowrk structure
    Processes the network topology and features of each drone to create node embeddings that caputre both local and network-wide information.

    Args:
        input_dim(int): number of input features per drone(default: 3)
            Features:
            - blocked status (0/1)
            - malicious process detection (0/1)
            - network events (0/1/2)
        hidden_dim(int): Dimension of hidden layers default is 64
    
    """
    def __init__(self, input_dim=5, hidden_dim=64):
        super(DroneGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        """Forward pass through the GNN.
        Args: x(tensor.Tesnor): Node feature matrix [num_nodes, input_dim]
            contains secruity state features for each drone
        edge_index(torch.Tensor): Graph connectivity in COO format [2, num_edges]
            Represents communication links between drones"""
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x