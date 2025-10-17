"""
Graph Neural Network Model for STL Classification
Implements a GNN architecture for binary classification of 3D models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional


class STLClassifier(nn.Module):
    """Graph Neural Network for STL file classification."""
    
    def __init__(self, 
                 input_dim: int = 9,  # Node feature dimension
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 pooling: str = 'mean'):
        """
        Initialize the STL classifier.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Whether to use attention-based GNN (GAT) or standard GCN
            pooling: Global pooling method ('mean', 'max', 'add', or 'concat')
        """
        super(STLClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                # Graph Attention Network
                if i == 0:
                    conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
                else:
                    conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            else:
                # Graph Convolutional Network
                conv = GCNConv(hidden_dim, hidden_dim)
            
            self.gnn_layers.append(conv)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling
        if pooling == 'concat':
            pool_dim = hidden_dim * 3  # mean + max + add
        else:
            pool_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # Binary classification
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the network."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            residual = x
            
            # Apply GNN layer
            if self.use_attention:
                x = gnn_layer(x, edge_index)
            else:
                x = gnn_layer(x, edge_index)
                x = F.relu(x)
            
            # Batch normalization
            x = bn(x)
            
            # Residual connection (if dimensions match)
            if x.shape == residual.shape:
                x = x + residual
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'concat':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_add], dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def predict(self, data: Data) -> torch.Tensor:
        """Make prediction on a single graph."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


