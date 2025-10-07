"""
Data Loading and Preprocessing Pipeline for STL Classification
Handles dataset loading, preprocessing, and batch creation for training.
"""

import os
import glob
import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pickle

from stl_processor import STLToGraphConverter


class STLDataset(Dataset):
    """Dataset class for STL files converted to graphs."""
    
    def __init__(self, 
                 data_dir: str,
                 max_vertices: int = 500,
                 max_edges: int = 2000,
                 cache_dir: Optional[str] = None,
                 force_rebuild: bool = False):
        """
        Initialize the STL dataset.
        
        Args:
            data_dir: Directory containing positive/negative subdirectories
            max_vertices: Maximum vertices per graph
            max_edges: Maximum edges per graph
            cache_dir: Directory to cache processed graphs
            force_rebuild: Force rebuild cache even if it exists
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        self.force_rebuild = force_rebuild
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize converter
        self.converter = STLToGraphConverter(
            max_vertices=max_vertices,
            max_edges=max_edges
        )
        
        # Load or build dataset
        self.graphs, self.labels = self._load_dataset()
        
        print(f"Loaded {len(self.graphs)} graphs")
        print(f"Positive samples: {sum(self.labels)}")
        print(f"Negative samples: {len(self.labels) - sum(self.labels)}")
    
    def _get_file_paths(self) -> Tuple[List[str], List[int]]:
        """Get all STL file paths and their labels."""
        file_paths = []
        labels = []
        
        # Positive samples (gun parts)
        positive_dir = os.path.join(self.data_dir, 'positive')
        if os.path.exists(positive_dir):
            positive_files = glob.glob(os.path.join(positive_dir, '*.stl')) + \
                           glob.glob(os.path.join(positive_dir, '*.STL'))
            file_paths.extend(positive_files)
            labels.extend([1] * len(positive_files))
        
        # Negative samples (non-gun parts)
        negative_dir = os.path.join(self.data_dir, 'negative')
        if os.path.exists(negative_dir):
            negative_files = glob.glob(os.path.join(negative_dir, '*.stl')) + \
                           glob.glob(os.path.join(negative_dir, '*.STL'))
            file_paths.extend(negative_files)
            labels.extend([0] * len(negative_files))
        
        return file_paths, labels
    
    def _load_dataset(self) -> Tuple[List[Data], List[int]]:
        """Load or build the dataset."""
        cache_file = os.path.join(self.cache_dir, 'dataset.pkl')
        
        # Check if cache exists and is valid
        if not self.force_rebuild and os.path.exists(cache_file):
            print("Loading cached dataset...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['graphs'], cached_data['labels']
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding...")
                
        
        # Build dataset from scratch
        print("Building dataset from STL files...")
        file_paths, labels = self._get_file_paths()
        
        if not file_paths:
            raise ValueError(f"No STL files found in {self.data_dir}")
        
        graphs = []
        valid_labels = []
        
        # Process files with progress bar
        for file_path, label in tqdm(zip(file_paths, labels), 
                                   total=len(file_paths), 
                                   desc="Converting STL files"):
            try:
                graph = self.converter.convert_stl_to_graph(file_path)
                if graph is not None:
                    graphs.append(graph)
                    valid_labels.append(label)
                else:
                    print(f"Failed to convert: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Cache the dataset
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'graphs': graphs,
                    'labels': valid_labels
                }, f)
            print(f"Dataset cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache dataset: {e}")
        
        return graphs, valid_labels
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        return self.graphs[idx], self.labels[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced dataset."""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weights inversely proportional to class frequency
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float)


def collate_fn(batch: List[Tuple[Data, int]]) -> Tuple[Batch, torch.Tensor]:
    """Custom collate function for batching graphs."""
    graphs, labels = zip(*batch)
    
    # Create batch from graphs
    batch_graph = Batch.from_data_list(graphs)
    
    # Convert labels to tensor
    batch_labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_graph, batch_labels


def create_data_loaders(data_dir: str,
                       batch_size: int = 8,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42,
                       **dataset_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for STLDataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = STLDataset(data_dir, **dataset_kwargs)
    
    # Split indices
    indices = list(range(len(full_dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, 
        stratify=full_dataset.labels
    )
    
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size, random_state=random_state,
        stratify=[full_dataset.labels[i] for i in train_indices]
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    return train_loader, val_loader, test_loader


def analyze_dataset(data_dir: str) -> dict:
    """Analyze the dataset and return statistics."""
    dataset = STLDataset(data_dir)
    
    # Basic statistics
    stats = {
        'total_samples': len(dataset),
        'positive_samples': sum(dataset.labels),
        'negative_samples': len(dataset.labels) - sum(dataset.labels),
        'class_balance': sum(dataset.labels) / len(dataset.labels)
    }
    
    # Graph statistics
    num_nodes = [graph.x.shape[0] for graph in dataset.graphs]
    num_edges = [graph.edge_index.shape[1] for graph in dataset.graphs]
    
    stats.update({
        'avg_nodes': np.mean(num_nodes),
        'std_nodes': np.std(num_nodes),
        'min_nodes': np.min(num_nodes),
        'max_nodes': np.max(num_nodes),
        'avg_edges': np.mean(num_edges),
        'std_edges': np.std(num_edges),
        'min_edges': np.min(num_edges),
        'max_edges': np.max(num_edges)
    })
    
    return stats


def test_data_loader():
    """Test the data loader functionality."""
    data_dir = "dataset"
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found")
        return
    
    print("Testing data loader...")
    
    # Test dataset creation
    try:
        dataset = STLDataset(data_dir, max_vertices=100, max_edges=500)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test single item
        graph, label = dataset[0]
        print(f"Sample graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        print(f"Sample label: {label}")
        
        # Test data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, batch_size=4, max_vertices=100, max_edges=500
        )
        
        # Test batch
        for batch_graph, batch_labels in train_loader:
            print(f"Batch: {batch_graph.x.shape[0]} total nodes, {len(batch_labels)} samples")
            print(f"Batch labels: {batch_labels.tolist()}")
            break
        
        # Analyze dataset
        stats = analyze_dataset(data_dir)
        print("\nDataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error testing data loader: {e}")


if __name__ == "__main__":
    test_data_loader()

