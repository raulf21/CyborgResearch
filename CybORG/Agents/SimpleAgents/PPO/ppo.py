import numpy as np
import torch
import torch.nn as nn
from CybORG.Agents.SimpleAgents.PPO.drone_gnn import DroneGNN


class PPONetwork(nn.Module):
    """PPO netowrk combining GNN processing with Actor-critic architecture.
    Processes the drone sware state using GNn and outputs action probabilites and state
    value estimates for PPO training.
    
    Args:
        num_drones (int): Number of drones in swarm. Default is 18
        output_dim (int): Number of actions. Default is 7:
        - RetakeControl: Remove red agents, Create blue agent
        - Remove other Sessions: Remove low priviledge sessions
        - BlockTraffic: Drop incoming traffic from target IP 
        - AllowTraffic: Strop dropping packets from target IP
        - rest defined in Appendix A    
    """
    def __init__(self, num_drones=18, output_dim=7):  # 7 actions from Appendix A
        super(PPONetwork, self).__init__()
        self.num_drones = num_drones
        
        # GNN for processing drone network structure
        self.gnn = DroneGNN(input_dim=5)  # 5 features per node from observation space
        
         # Actor network: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
       # Critic network: estimates state value
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def create_graph_data(self, state):
        """Converts raw state observation into graph structure.
        
        Args:
            state(np.ndarray): Raw observation from env.
                Format defined in Appendix B
        returns:
            tuple: (node_features, edge_index)
                - node_features(torch.Tensor): [num_nodes, 3] feature matrix
                - edge_index (torch.Tensor): [2, num_edges] connectivity matrix"""
        if len(state.shape) > 1:
            state = state[0]
            
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
            
        node_features = []
        edge_index = []
        
        # Extract node features for each drone based on Appendix B
        for i in range(self.num_drones):
            # Get blocked status (index 1+i)
            blocked_status = float(state[1 + i])
            
            # Get malicious process indicator (index n+1)
            malicious_process = float(state[self.num_drones + 1])
            
            # Get network events (index n+2+i)
            network_events = float(state[self.num_drones + 2 + i])

            one_hot_network_events = [0.0, 0.0, 0.0]
            
            one_hot_network_events[int(network_events)] = 1.0
            
            node_features.append([blocked_status, malicious_process] + one_hot_network_events)
        
        # Create edges based on drone positions and 30-unit communication range
        for i in range(self.num_drones):
            x_i = float(state[2*self.num_drones + 2 + 2*i])
            y_i = float(state[2*self.num_drones + 2 + 2*i + 1])
            
            for j in range(i+1, self.num_drones):
                x_j = float(state[2*self.num_drones + 2 + 2*j])
                y_j = float(state[2*self.num_drones + 2 + 2*j + 1])
                
                # Create edge if drones are within communication range (30 units)
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                if dist < 30.0:
                    edge_index.extend([[i, j], [j, i]])
        
        # Add self-loops if no edges exist
        if not edge_index:
            edge_index = [[i, i] for i in range(self.num_drones)]
            
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        
        return x, edge_index

    def forward(self, state):
        """ Processes state throug GNN and output action proabilities and value.
        
        Args:
            state (np.ndarray): Raw observation from environemnt
        Returns:
            tuple: (action_probs, value)
                - action_probs (torch.Tensor): Action probability distribution
                - value (torch.Tensor): Estimated state value"""
        x, edge_index = self.create_graph_data(state)
        node_embeddings = self.gnn(x, edge_index)
        # Use first node's embedding for decision making
        agent_embedding = node_embeddings[0]
        action_probs = self.actor(agent_embedding)
        value = self.critic(agent_embedding)
        return action_probs, value