import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class GNNApplication(nn.Module):
    def __init__(self, input_dim, hidden_dim_gnn, output_dim):
        super(GNNApplication, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim_gnn)
        self.conv2 = GCNConv(hidden_dim_gnn, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, data.batch)
        return x


class GNNCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim_gnn, embedding_dim, attention_dim, output_dim, num_nodes):
        super(GNNCritic, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(input_dim, hidden_dim_gnn)
        self.conv2 = GCNConv(hidden_dim_gnn, hidden_dim_gnn) # (batch_size, num_nodes, output_dim)
        self.relu = nn.ReLU()
        
        self.attention_fc = nn.Linear(hidden_dim_gnn + embedding_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)

        self.output_fc = nn.Linear(hidden_dim_gnn, output_dim)  # Adjust the output size as needed
        
    def forward(self, data, app_embedding):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()

        # print("x.shape", x.shape)
        # print("data.batch.shape", data.batch.shape)
        # print("app_embedding.shape", app_embedding.shape)
        
        expanded_embedding_vectors = app_embedding[data.batch].squeeze(1)  # Shape: [num_nodes, embedding_dim]

        # print("expanded_embedding_vectors.shape", expanded_embedding_vectors.shape)
        
        combined_embedding = torch.cat([x, expanded_embedding_vectors], dim=-1)  # Shape: [num_nodes, gnn_hidden_dim + embedding_dim]

        # print("combined_embedding.shape", combined_embedding.shape)

        # Attention mechanism and normalization
        attention_hidden = self.attention_fc(combined_embedding) # Linear transformation
        attention_scores = F.leaky_relu(attention_hidden, negative_slope=0.2)  # LeakyReLU activation
        attention_weights = torch.softmax(self.attention_score(attention_scores), dim=0)

        # Apply attention weights to the node embeddings
        attention_output = attention_weights * x

        # Final output per node
        node_output = self.output_fc(attention_output)
        
        batch_size = app_embedding.size(0)  # Number of graphs in the batch
        output_per_graph = node_output.view(batch_size, self.num_nodes, -1)  # Reshape to [batch_size, num_nodes, output_dim]
        
        # Flatten each graph's output to [batch_size, num_nodes * output_dim]
        flattened_output = output_per_graph.view(batch_size, -1)  # Shape: [batch_size, num_nodes * output_dim]
        flattened_output = torch.flatten(flattened_output)
        state_val = global_mean_pool(flattened_output, data.batch)
        
        return state_val  # Shape: [batch_size, 1]


class GNNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim_gnn, embedding_dim, attention_dim, output_dim, num_nodes):
        super(GNNAttention, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(input_dim, hidden_dim_gnn)
        self.conv2 = GCNConv(hidden_dim_gnn, hidden_dim_gnn) # (batch_size, num_nodes, output_dim)
        self.relu = nn.ReLU()
        
        self.attention_fc = nn.Linear(hidden_dim_gnn + embedding_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)

        self.output_fc = nn.Linear(hidden_dim_gnn, output_dim)  # Adjust the output size as needed

    def forward(self, data, app_embedding):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()

        expanded_embedding_vectors = app_embedding[data.batch]
        combined_embedding = torch.cat([x, expanded_embedding_vectors], dim=-1)  # Shape: [num_nodes, gnn_hidden_dim + embedding_dim]

        # Attention mechanism and normalization
        attention_hidden = self.attention_fc(combined_embedding) # Linear transformation
        attention_scores = F.leaky_relu(attention_hidden, negative_slope=0.2)  # LeakyReLU activation
        attention_weights = torch.softmax(self.attention_score(attention_scores), dim=0)

        # Apply attention weights to the node embeddings
        attention_output = attention_weights * x

        # Final output per node
        node_output = self.output_fc(attention_output)
        
        batch_size = app_embedding.size(0)  # Number of graphs in the batch
        output_per_graph = node_output.view(batch_size, self.num_nodes, -1)  # Reshape to [batch_size, num_nodes, output_dim]
        
        # Flatten each graph's output to [batch_size, num_nodes * output_dim]
        flattened_output = output_per_graph.view(batch_size, -1)  # Shape: [batch_size, num_nodes * output_dim]

        return flattened_output  # Shape: [batch_size, num_nodes * output_dim]
    