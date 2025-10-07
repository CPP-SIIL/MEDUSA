"""
Quick test script to verify the STL classification system works
Tests with a small subset of files for faster verification.
"""

import os
import torch
from stl_processor import STLToGraphConverter
from gnn_model import SimpleSTLClassifier
from data_loader import STLDataset
import glob

def test_single_file():
    """Test conversion of a single STL file."""
    print("Testing single STL file conversion...")
    
    # Find a sample file
    sample_files = glob.glob("dataset/positive/*.STL")[:1]
    if not sample_files:
        sample_files = glob.glob("dataset/positive/*.stl")[:1]
    
    if not sample_files:
        print("No STL files found in dataset/positive/")
        return False
    
    sample_file = sample_files[0]
    print(f"Testing with: {sample_file}")
    
    # Test converter
    converter = STLToGraphConverter(max_vertices=100, max_edges=500)
    graph = converter.convert_stl_to_graph(sample_file)
    
    if graph is None:
        print("Failed to convert STL to graph")
        return False
    
    print(f"[OK] Graph created: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
    print(f"[OK] Node features: {graph.x.shape[1]} dimensions")
    print(f"[OK] Edge attributes: {graph.edge_attr.shape[1]} dimensions")
    
    return True

def test_model():
    """Test the GNN model with dummy data."""
    print("\nTesting GNN model...")
    
    # Create dummy data
    num_nodes = 50
    num_features = 9
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test model
    model = SimpleSTLClassifier(input_dim=num_features, hidden_dim=16)
    output = model(data)
    
    print(f"[OK] Model output shape: {output.shape}")
    print(f"[OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test prediction
    pred, prob = model.predict(data)
    print(f"[OK] Prediction: {pred.item()}, Probability: {prob[0].tolist()}")
    
    return True

def test_small_dataset():
    """Test with a very small dataset."""
    print("\nTesting small dataset...")
    
    # Create a small dataset with just a few files
    try:
        dataset = STLDataset(
            "dataset", 
            max_vertices=50, 
            max_edges=200,
            force_rebuild=True  # Force rebuild for testing
        )
        
        # Limit to first 5 files for quick test
        if len(dataset) > 5:
            dataset.graphs = dataset.graphs[:5]
            dataset.labels = dataset.labels[:5]
        
        print(f"[OK] Dataset created with {len(dataset)} samples")
        
        # Test single item
        graph, label = dataset[0]
        print(f"[OK] Sample graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")
        print(f"[OK] Sample label: {label}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running quick tests for STL classification system...\n")
    
    tests = [
        ("Single file conversion", test_single_file),
        ("GNN model", test_model),
        ("Small dataset", test_small_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with error: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("SUCCESS: All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python train.py --model_type simple --epochs 10")
    else:
        print("ERROR: Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
