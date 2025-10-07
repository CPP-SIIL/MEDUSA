"""
STL to Graph Converter for 3D Model Classification
Converts STL files to graph representations suitable for GNN training.
"""

import numpy as np
import trimesh
import networkx as nx
from typing import Tuple, List
import torch
from torch_geometric.data import Data
import os


class STLToGraphConverter:
    """Converts STL files to graph representations for GNN processing."""
    
    def __init__(self, 
                 max_vertices: int = 1000,
                 max_edges: int = 5000,
                 sampling_method: str = 'uniform'):
        """
        Initialize the STL to graph converter.
        
        Args:
            max_vertices: Maximum number of vertices to sample from the mesh
            max_edges: Maximum number of edges to include in the graph
            sampling_method: Method for sampling vertices ('uniform' or 'poisson')
        """
        self.max_vertices = max_vertices
        self.max_edges = max_edges
        self.sampling_method = sampling_method
    
    def load_stl(self, file_path: str) -> trimesh.Trimesh:
        """Load STL file and return trimesh object."""
        try:
            mesh = trimesh.load(file_path)
            if not isinstance(mesh, trimesh.Trimesh):
                # Handle scene objects
                if hasattr(mesh, 'geometry'):
                    mesh = list(mesh.geometry.values())[0]
                else:
                    raise ValueError("Could not extract mesh from file")
            return mesh
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def sample_vertices(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Sample vertices from the mesh."""
        if self.sampling_method == 'uniform':
            # Uniform sampling on the mesh surface
            points = mesh.sample(self.max_vertices)
        elif self.sampling_method == 'poisson':
            # Poisson disk sampling for more uniform distribution
            points = mesh.sample(self.max_vertices)
        else:
            # Use existing vertices if mesh is small enough
            if len(mesh.vertices) <= self.max_vertices:
                points = mesh.vertices
            else:
                # Randomly sample existing vertices
                indices = np.random.choice(len(mesh.vertices), 
                                        self.max_vertices, replace=False)
                points = mesh.vertices[indices]
        
        return points
    
    def compute_features(self, points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute node features for each sampled point."""
        features = []
        
        for point in points:
            # Basic geometric features
            feature_vector = []
            
            # 1. 3D coordinates (normalized)
            feature_vector.extend(point)
            
            # 2. Distance to centroid
            centroid = mesh.centroid
            dist_to_centroid = np.linalg.norm(point - centroid)
            feature_vector.append(dist_to_centroid)
            
            # 3. Distance to bounding box center
            bbox_center = mesh.bounds.mean(axis=0)
            dist_to_bbox_center = np.linalg.norm(point - bbox_center)
            feature_vector.append(dist_to_bbox_center)
            
            # 4. Local surface normal (approximate)
            # Find closest face and use its normal
            try:
                # Find closest vertex and use its face normal
                distances = np.linalg.norm(mesh.vertices - point, axis=1)
                closest_vertex_idx = np.argmin(distances)
                
                # Find faces that contain this vertex
                face_indices = np.where(np.any(mesh.faces == closest_vertex_idx, axis=1))[0]
                if len(face_indices) > 0:
                    # Use the first face's normal
                    normal = mesh.face_normals[face_indices[0]]
                    feature_vector.extend(normal)
                else:
                    feature_vector.extend([0, 0, 1])  # Default normal
            except:
                feature_vector.extend([0, 0, 1])  # Default normal
            
            # 5. Curvature approximation (simplified)
            # Use distance to nearest neighbors as curvature proxy
            distances = np.linalg.norm(points - point, axis=1)
            distances = np.sort(distances)[1:6]  # Skip self, take 5 nearest
            curvature_proxy = np.std(distances)
            feature_vector.append(curvature_proxy)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def build_graph(self, points: np.ndarray, features: np.ndarray) -> nx.Graph:
        """Build a graph from sampled points and features."""
        G = nx.Graph()
        
        # Add nodes with features
        for i, (point, feature) in enumerate(zip(points, features)):
            G.add_node(i, pos=point, features=feature)
        
        # Add edges based on spatial proximity
        # Use k-nearest neighbors approach
        k = min(8, len(points) - 1)  # Connect to k nearest neighbors
        
        for i, point in enumerate(points):
            distances = np.linalg.norm(points - point, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # Skip self
            
            for j in nearest_indices:
                if not G.has_edge(i, j):
                    edge_weight = 1.0 / (distances[j] + 1e-6)  # Inverse distance weight
                    G.add_edge(i, j, weight=edge_weight)
        
        return G
    
    def graph_to_pytorch_geometric(self, graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        # Extract node features
        node_features = []
        node_positions = []
        
        for node in graph.nodes():
            features = graph.nodes[node]['features']
            pos = graph.nodes[node]['pos']
            node_features.append(features)
            node_positions.append(pos)
        
        node_features = np.array(node_features)
        node_positions = np.array(node_positions)
        
        # Extract edge information
        edge_list = []
        edge_weights = []
        
        for edge in graph.edges():
            edge_list.append([edge[0], edge[1]])
            edge_list.append([edge[1], edge[0]])  # Add reverse edge for undirected graph
            weight = graph.edges[edge]['weight']
            edge_weights.extend([weight, weight])
        
        if len(edge_list) == 0:
            # Handle isolated nodes
            edge_list = [[0, 0]]  # Self-loop for single node
            edge_weights = [1.0]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(node_positions, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    def convert_stl_to_graph(self, file_path: str) -> Data:
        """Convert STL file to PyTorch Geometric Data object."""
        # Load mesh
        mesh = self.load_stl(file_path)
        if mesh is None:
            return None
        
        # Sample vertices
        points = self.sample_vertices(mesh)
        
        # Compute features
        features = self.compute_features(points, mesh)
        
        # Build graph
        graph = self.build_graph(points, features)
        
        # Convert to PyTorch Geometric format
        data = self.graph_to_pytorch_geometric(graph)
        
        return data
    
    def batch_convert(self, file_paths: List[str]) -> List[Data]:
        """Convert multiple STL files to graphs."""
        graphs = []
        for file_path in file_paths:
            graph = self.convert_stl_to_graph(file_path)
            if graph is not None:
                graphs.append(graph)
        return graphs


def test_converter():
    """Test the STL to graph converter with a sample file."""
    converter = STLToGraphConverter(max_vertices=500, max_edges=2000)
    
    # Test with a sample file from the dataset
    sample_file = "dataset/positive/barrel.STL"
    if os.path.exists(sample_file):
        print(f"Testing conversion of {sample_file}")
        graph_data = converter.convert_stl_to_graph(sample_file)
        
        if graph_data is not None:
            print(f"Graph created successfully!")
            print(f"Number of nodes: {graph_data.x.shape[0]}")
            print(f"Number of edges: {graph_data.edge_index.shape[1]}")
            print(f"Node feature dimension: {graph_data.x.shape[1]}")
            print(f"Edge attribute dimension: {graph_data.edge_attr.shape[1]}")
        else:
            print("Failed to create graph")
    else:
        print(f"Sample file {sample_file} not found")


if __name__ == "__main__":
    test_converter()
