# MEDUSA
### Machine-learning Engine for Detecting Unlawful Shapes Automatically

This project implements a Graph Neural Network (GNN) for classifying STL 3D models into two categories: gun parts or non-gun parts.

## Overview

The system converts 3D STL models into graph representations and uses a Graph Neural Network to perform binary classification. The approach involves:

1. **STL Processing**: Converting 3D meshes to graph structures with sampled vertices and edges
2. **Feature Extraction**: Computing geometric features for each vertex (position, normals, curvature, etc.)
3. **Graph Neural Network**: Using GCN/GAT layers for learning graph representations
4. **Classification**: Binary classification with proper handling of imbalanced datasets

## Dataset

The dataset contains approximately 300 STL files:
- **Positive class**: ~100 gun part STL files (barrels, frames, slides, triggers, etc.)
- **Negative class**: ~200 non-gun part STL files (various mechanical parts)

Files are organized in:
```
dataset/
├── positive/     # Gun parts
└── negative/     # Non-gun parts
```

## Installation

1. Make sure you have a Python virtual environment activated
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Train a simple model with default settings:
```bash
python train.py --model_type simple --epochs 30
```

### Advanced Training

Train with custom parameters:
```bash
python train.py \
    --model_type full \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 64 \
    --max_vertices 1000 \
    --patience 20
```

### Command Line Arguments

- `--data_dir`: Dataset directory (default: 'dataset')
- `--model_type`: Model type - 'simple' or 'full' (default: 'simple')
- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--hidden_dim`: Hidden dimension size (default: 32)
- `--max_vertices`: Maximum vertices per graph (default: 500)
- `--max_edges`: Maximum edges per graph (default: 2000)
- `--patience`: Early stopping patience (default: 15)
- `--output_dir`: Output directory for results (default: 'outputs')

## Model Architecture

### Simple Model
- 3 GCN layers with ReLU activation
- Global mean pooling
- 2-layer MLP classifier

### Full Model
- Graph Attention Network (GAT) or GCN layers
- Batch normalization and residual connections
- Multiple pooling strategies (mean, max, add, concat)
- Deeper classification head with dropout

## File Structure

```
├── stl_processor.py    # STL to graph conversion
├── gnn_model.py        # GNN model definitions
├── data_loader.py      # Dataset loading and preprocessing
├── train.py           # Main training script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Key Features

- **Automatic STL Processing**: Converts 3D meshes to graph representations
- **Geometric Features**: Extracts meaningful features from 3D geometry
- **Imbalanced Dataset Handling**: Uses class weights and proper evaluation metrics
- **Caching**: Caches processed graphs for faster subsequent runs
- **Visualization**: Generates training plots and confusion matrices
- **TensorBoard Integration**: Logs training metrics for monitoring
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Model Checkpointing**: Saves best model based on validation loss

## Output

The training process generates:
- `outputs/results.json`: Training and test metrics
- `outputs/model.pth`: Saved model weights and configuration
- `outputs/training_history.png`: Training curves
- TensorBoard logs in `runs/` directory

## Performance Considerations

- **Memory Usage**: Larger graphs (more vertices/edges) require more memory
- **Processing Time**: STL conversion is the most time-consuming step
- **Caching**: First run processes all STL files; subsequent runs use cache
- **Batch Size**: Adjust based on available GPU memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or max_vertices
2. **STL Loading Errors**: Some STL files may be corrupted or in unsupported format
3. **Slow Processing**: Enable caching by not using `--force_rebuild`

### Windows Compatibility

The code is designed to work on Windows with:
- `num_workers=0` in DataLoader for Windows compatibility
- Proper path handling for Windows file systems

## Future Improvements

- Support for more 3D file formats (OBJ, PLY, etc.)
- Advanced geometric features (curvature, shape descriptors)
- Data augmentation techniques for 3D graphs
- Ensemble methods for improved accuracy
- Real-time inference pipeline

