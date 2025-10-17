"""
Flask Server for STL Classification Demo
Web-based interface for real-time GNN model inference and visualization
"""

import os
import sys
import json
import glob
from typing import Dict, List, Optional, Tuple, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_model import STLClassifier
from stl_processor import STLToGraphConverter

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class STLInferenceServer:
    """Main server class for handling STL model inference and management."""
    
    def __init__(self, model_folder: str = None, examples_dir: str = r"C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Fall 2025\demo\examples"):
        """
        Initialize the inference server.
        
        Args:
            model_folder: Path to trained model folder
            examples_dir: Directory containing demo STL files
        """
        self.examples_dir = examples_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        if model_folder:
            self.model, self.model_config = self._load_model(model_folder)
        else:
            self.model = None
            self.model_config = {}
        
        # Initialize STL processor
        self.stl_processor = STLToGraphConverter(max_vertices=500, max_edges=2000)
        
        # Cache for model metadata
        self._model_cache = {}
        self._build_model_cache()
        
        print(f"Server initialized with {len(self._model_cache)} demo models")
        print(f"Device: {self.device}")
        if self.model:
            print(f"Model loaded: {self.model_config.get('model_type', 'unknown')}")
    
    def _load_model(self, model_folder: str) -> Tuple[torch.nn.Module, Dict]:
        """Load the trained model from folder."""
        model_path = os.path.join(model_folder, 'model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('model_config', {})
        
        # Create model
        model = STLClassifier(
            input_dim=model_config.get('input_dim', 9),
            hidden_dim=model_config.get('hidden_dim', 64)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, model_config
    
    def _build_model_cache(self):
        """Build cache of available demo models with metadata."""
        self._model_cache = {}
        
        for class_type in ['positive', 'negative']:
            class_dir = os.path.join(self.examples_dir, class_type)
            if os.path.exists(class_dir):
                for file_path in glob.glob(os.path.join(class_dir, '*.stl')) + glob.glob(os.path.join(class_dir, '*.STL')):
                    filename = os.path.basename(file_path)
                    name = os.path.splitext(filename)[0]
                    
                    # Extract confidence from filename if present
                    confidence = None
                    if '_conf' in name:
                        try:
                            conf_part = name.split('_conf')[1]
                            confidence = float(conf_part)
                            # Clean name by removing confidence suffix
                            name = name.split('_conf')[0]
                        except (ValueError, IndexError):
                            pass
                    
                    model_id = f"{class_type}_{filename}"
                    self._model_cache[model_id] = {
                        'id': model_id,
                        'name': name,
                        'filename': filename,
                        'type': class_type,
                        'path': file_path,
                        'url': f"/models/{class_type}/{filename}",
                        'thumbnail': f"/thumbnails/{class_type}/{filename}.png",
                        'confidence': confidence,
                        'expected_result': 1 if class_type == 'positive' else 0
                    }
    
    def get_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """
        Get list of available 3D models.
        
        Args:
            model_type: Filter by 'positive' or 'negative', None for all
            
        Returns:
            List of model metadata dictionaries
        """
        models = list(self._model_cache.values())
        
        if model_type:
            models = [m for m in models if m['type'] == model_type]
        
        # Sort by confidence (highest first) then by name
        models.sort(key=lambda x: (-(x.get('confidence') or 0), x['name']))
        
        return models
    
    def extract_layer_activations(self, graph_data) -> Dict[str, Any]:
        """
        Extract simplified layer activations for visualization.
        
        Args:
            graph_data: Input graph data
            
        Returns:
            Dictionary with layer activations
        """
        if not self.model:
            return {}
        
        activations = {}
        layer_outputs = []
        
        # Hook function to capture intermediate outputs
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Aggregate tensor to manageable size for visualization
                    aggregated = self._aggregate_tensor(output, name)
                    activations[name] = aggregated
            return hook
        
        # Register hooks for different layer types
        hooks = []
        layer_count = 0
        
        for name, module in self.model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['conv', 'gnn', 'gcn', 'gat', 'linear', 'classifier']):
                hook = module.register_forward_hook(hook_fn(f"layer_{layer_count}_{name}"))
                hooks.append(hook)
                layer_count += 1
        
        # Run inference to capture activations
        with torch.no_grad():
            _ = self.model(graph_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Structure activations for frontend
        structured_activations = {}
        for i, (name, activation) in enumerate(activations.items()):
            structured_activations[f"layer_{i}"] = {
                'name': name,
                'values': activation,
                'type': self._get_layer_type(name),
                'size': len(activation) if isinstance(activation, list) else 1
            }
        
        return structured_activations
    
    def _aggregate_tensor(self, tensor: torch.Tensor, layer_name: str) -> List[float]:
        """
        Aggregate tensor values for visualization.
        
        Args:
            tensor: Input tensor
            layer_name: Name of the layer
            
        Returns:
            List of aggregated values
        """
        if tensor.dim() == 0:
            return [tensor.item()]
        
        # Flatten and sample/aggregate the tensor
        flat = tensor.flatten()
        
        # Limit to reasonable size for frontend
        max_values = 50
        
        if len(flat) <= max_values:
            return flat.cpu().tolist()
        else:
            # Sample evenly across the tensor
            indices = np.linspace(0, len(flat) - 1, max_values, dtype=int)
            sampled = flat[indices]
            return sampled.cpu().tolist()
    
    def _get_layer_type(self, layer_name: str) -> str:
        """Determine layer type from name."""
        name_lower = layer_name.lower()
        if 'conv' in name_lower or 'gcn' in name_lower:
            return 'convolution'
        elif 'gat' in name_lower:
            return 'attention'
        elif 'linear' in name_lower or 'classifier' in name_lower:
            return 'dense'
        elif 'pool' in name_lower:
            return 'pooling'
        else:
            return 'other'
    
    def perform_inference(self, model_url: str) -> Dict[str, Any]:
        """
        Perform inference on a 3D model.
        
        Args:
            model_url: URL/path to the 3D model
            
        Returns:
            Dictionary with inference results and weight data
        """
        if not self.model:
            return {'error': 'No model loaded'}
        
        try:
            # Find model file path from URL
            model_path = None
            for model_data in self._model_cache.values():
                if model_data['url'] == model_url:
                    model_path = model_data['path']
                    break
            
            if not model_path or not os.path.exists(model_path):
                return {'error': f'Model file not found: {model_url}'}
            
            # Convert STL to graph
            graph = self.stl_processor.convert_stl_to_graph(model_path)
            if graph is None:
                return {'error': 'Failed to process STL file'}
            
            # Prepare graph for inference
            graph = graph.to(self.device)
            batch_graph = Batch.from_data_list([graph])
            
            # Perform inference and extract activations
            start_time = time.time()
            
            with torch.no_grad():
                # Get model output
                output = self.model(batch_graph)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
                
                # Extract layer activations for visualization
                activations = self.extract_layer_activations(batch_graph)
            
            inference_time = time.time() - start_time
            
            # Prepare response
            result = {
                'prediction': {
                    'class': predicted_class,
                    'confidence': float(confidence),
                    'probabilities': probabilities[0].cpu().tolist(),
                    'labels': ['Not Gun Part', 'Gun Part']
                },
                'weights': activations,
                'metadata': {
                    'inference_time': inference_time,
                    'model_type': self.model_config.get('model_type', 'unknown'),
                    'graph_info': {
                        'num_nodes': graph.x.shape[0],
                        'num_edges': graph.edge_index.shape[1],
                        'num_features': graph.x.shape[1]
                    }
                }
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Inference failed: {str(e)}'}

# Initialize the server
server = STLInferenceServer()

# API Endpoints
@app.route('/api/ping', methods=['GET'])
def ping():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'model_loaded': server.model is not None,
        'device': str(server.device)
    })

@app.route('/api/getModels', methods=['GET'])
def get_models():
    """Get list of available 3D models."""
    model_type = request.args.get('type')
    
    try:
        models = server.get_models(model_type)
        return jsonify({
            'success': True,
            'models': models,
            'total': len(models)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/doInference', methods=['POST'])
def do_inference():
    """Perform inference on a 3D model."""
    data = request.get_json()
    
    if not data or 'modelUrl' not in data:
        return jsonify({
            'success': False,
            'error': 'modelUrl parameter required'
        }), 400
    
    model_url = data['modelUrl']
    
    try:
        result = server.perform_inference(model_url)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Inference error: {str(e)}'
        }), 500

# File serving endpoints
@app.route('/models/<path:filename>')
def serve_model_file(filename):
    """Serve STL model files."""
    # Handle both positive and negative subdirectories
    for class_type in ['positive', 'negative']:
        class_dir = os.path.join(server.examples_dir, class_type)
        file_path = os.path.join(class_dir, filename)
        if os.path.exists(file_path):
            return send_from_directory(class_dir, filename)
    
    return "File not found", 404

@app.route('/models/<class_type>/<path:filename>')
def serve_model_file_by_type(class_type, filename):
    """Serve STL model files by type."""
    class_dir = os.path.join(server.examples_dir, class_type)
    if os.path.exists(os.path.join(class_dir, filename)):
        return send_from_directory(class_dir, filename)
    
    return "File not found", 404

@app.route('/thumbnails/<class_type>/<path:filename>')
def serve_thumbnail(class_type, filename):
    """Serve thumbnail PNG images for demo models.

    This will look for thumbnails in several locations (in order):
    - Next to the STL file: demo/examples/<class_type>/<base>.png
    - Thumbnails sibling folder: demo/examples/<class_type>/<base>.stl.png
    - Central thumbnails dir: demo/examples/thumbnails/<class_type>/<filename>

    The frontend requests URLs like `/thumbnails/positive/model.stl.png` or
    `/thumbnails/positive/model.stl.png`. We tolerate both bare names and
    names that include the `.stl` suffix.
    """
    # Normalize class_type
    if class_type not in ('positive', 'negative'):
        return "Invalid class type", 400

    # Possible candidate paths (in order)
    examples_root = server.examples_dir
    candidates = []

    # If the requested filename already ends with .png, use it directly
    if filename.lower().endswith('.png'):
        candidates.append(os.path.join(examples_root, class_type, filename))
        # Also consider a sibling 'thumbnails' folder
        candidates.append(os.path.join(examples_root, 'thumbnails', class_type, filename))
        # If filename is like 'name.stl.png', also try 'name.png' (strip the .stl)
        if filename.lower().endswith('.stl.png'):
            stripped = filename[:-len('.stl.png')] + '.png'
            candidates.append(os.path.join(examples_root, class_type, stripped))
            candidates.append(os.path.join(examples_root, 'thumbnails', class_type, stripped))
    else:
        # Try filename.png next to the STL
        candidates.append(os.path.join(examples_root, class_type, f"{filename}.png"))
        # Try filename with .stl.png suffix
        candidates.append(os.path.join(examples_root, class_type, f"{filename}.stl.png"))
        # Try central thumbnails folder
        candidates.append(os.path.join(examples_root, 'thumbnails', class_type, f"{filename}.png"))
        candidates.append(os.path.join(examples_root, 'thumbnails', class_type, f"{filename}.stl.png"))

    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            # Serve from the directory containing the file
            dirpath = os.path.dirname(abs_path)
            fname = os.path.basename(abs_path)
            return send_from_directory(dirpath, fname)

    # Not found - print debug info for troubleshooting
    debug_list = []
    for p in candidates:
        debug_list.append({
            'candidate': p,
            'abs': os.path.abspath(p),
            'exists': os.path.exists(os.path.abspath(p))
        })
    print(f"[thumbnail] not found for {class_type}/{filename}, candidates:\n{json.dumps(debug_list, indent=2)}")
    return jsonify({'error': 'Thumbnail not found', 'candidates': debug_list}), 404

@app.route('/')
def index():
    """Serve a simple API documentation page."""
    return """
    <h1>STL Classification API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><strong>GET /api/ping</strong> - Health check</li>
        <li><strong>GET /api/getModels?type={positive|negative}</strong> - Get available models</li>
        <li><strong>POST /api/doInference</strong> - Perform inference (JSON body: {"modelUrl": "..."})</li>
    </ul>
    <h2>File Endpoints:</h2>
    <ul>
        <li><strong>GET /models/{positive|negative}/{filename}</strong> - Download STL files</li>
        <li><strong>GET /thumbnails/{positive|negative}/{filename}</strong> - Get thumbnails</li>
    </ul>
    """

if __name__ == '__main__':
    # Find the latest model automatically
    model_folder = r'C:\Users\awebb\Documents\Programming\Work\SIIL\Cyber Fair Fall 2025\outputs\6_1759440938.2232356'
    outputs_dir = "../outputs"
    
    if os.path.exists(outputs_dir):
        model_dirs = [d for d in os.listdir(outputs_dir) 
                     if os.path.isdir(os.path.join(outputs_dir, d)) and 
                     os.path.exists(os.path.join(outputs_dir, d, "model.pth"))]
        
        if model_dirs:
            # Sort by creation time (latest first)
            model_dirs.sort(key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)), reverse=True)
            model_folder = os.path.join(outputs_dir, model_dirs[0])
            print(f"Using latest model: {model_folder}")
    
    if model_folder and os.path.exists(model_folder):
        server = STLInferenceServer(model_folder)
    else:
        print(f"Warning: No valid model found. Server will run without model.")
        server = STLInferenceServer()
    
    # Run the server
    app.run(debug=False, host='0.0.0.0', port=5000)
