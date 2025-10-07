"""
Utility Functions for Demo Application
Common utilities and helper functions for the demo.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting functions will be disabled.")


def load_selection_report(report_path: str = "demo/selection_report.json") -> Dict[str, Any]:
    """Load the selection report JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)


def print_model_summary(report: Dict[str, Any]) -> None:
    """Print a formatted summary of the model and selection results."""
    print("=== MODEL SUMMARY ===")
    model_info = report['model_info']
    dataset_info = report['dataset_info']
    performance = report['model_performance']
    
    print(f"Model Type: {model_info['model_type']}")
    print(f"Input Dimension: {model_info['input_dim']}")
    print(f"Hidden Dimension: {model_info['hidden_dim']}")
    print(f"Model Folder: {model_info['model_folder']}")
    
    print(f"\nDataset: {dataset_info['total_samples']} total samples")
    print(f"  Positive: {dataset_info['positive_samples']}")
    print(f"  Negative: {dataset_info['negative_samples']}")
    
    print(f"\nModel Performance:")
    print(f"  Overall Accuracy: {performance['overall_accuracy']:.3f}")
    print(f"  Correct Predictions: {performance['total_correct']}/{performance['total_samples']}")


def analyze_confidence_distribution(report: Dict[str, Any]) -> None:
    """Analyze and plot confidence distribution of selected examples."""
    positive_examples = report['selected_examples']['positive']
    negative_examples = report['selected_examples']['negative']
    
    if not positive_examples and not negative_examples:
        print("No examples found for analysis.")
        return
    
    # Extract confidence scores
    pos_confidences = [ex['confidence'] for ex in positive_examples]
    neg_confidences = [ex['confidence'] for ex in negative_examples]
    
    # Print statistics
    print("\n=== CONFIDENCE ANALYSIS ===")
    if pos_confidences:
        print(f"Positive Examples: {len(pos_confidences)}")
        print(f"  Mean Confidence: {np.mean(pos_confidences):.3f}")
        print(f"  Min Confidence: {np.min(pos_confidences):.3f}")
        print(f"  Max Confidence: {np.max(pos_confidences):.3f}")
    
    if neg_confidences:
        print(f"Negative Examples: {len(neg_confidences)}")
        print(f"  Mean Confidence: {np.mean(neg_confidences):.3f}")
        print(f"  Min Confidence: {np.min(neg_confidences):.3f}")
        print(f"  Max Confidence: {np.max(neg_confidences):.3f}")


def create_confidence_plot(report: Dict[str, Any], save_path: str = "demo/confidence_distribution.png") -> None:
    """Create and save a confidence distribution plot."""
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plot creation.")
        return
        
    positive_examples = report['selected_examples']['positive']
    negative_examples = report['selected_examples']['negative']
    
    pos_confidences = [ex['confidence'] for ex in positive_examples]
    neg_confidences = [ex['confidence'] for ex in negative_examples]
    
    if not pos_confidences and not neg_confidences:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.subplot(1, 2, 1)
    if pos_confidences:
        plt.hist(pos_confidences, bins=10, alpha=0.7, label='Positive', color='green')
    if neg_confidences:
        plt.hist(neg_confidences, bins=10, alpha=0.7, label='Negative', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot box plots
    plt.subplot(1, 2, 2)
    data_to_plot = []
    labels = []
    if pos_confidences:
        data_to_plot.append(pos_confidences)
        labels.append('Positive')
    if neg_confidences:
        data_to_plot.append(neg_confidences)
        labels.append('Negative')
    
    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.ylabel('Confidence Score')
        plt.title('Confidence Box Plot')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence plot saved to: {save_path}")
    plt.close()


def list_demo_files() -> Dict[str, List[str]]:
    """List all files in the demo directories."""
    demo_files = {'positive': [], 'negative': []}
    
    for class_name in ['positive', 'negative']:
        class_dir = f"demo/examples/{class_name}"
        if os.path.exists(class_dir):
            demo_files[class_name] = [f for f in os.listdir(class_dir) if f.endswith('.stl') or f.endswith('.STL')]
            demo_files[class_name].sort()
    
    return demo_files


def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get information about a file."""
    if not os.path.exists(filepath):
        return {'exists': False}
    
    stat = os.stat(filepath)
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'filename': os.path.basename(filepath),
        'directory': os.path.dirname(filepath)
    }


def validate_demo_setup() -> bool:
    """Validate that the demo setup is complete and correct."""
    print("=== VALIDATING DEMO SETUP ===")
    
    # Check directories
    required_dirs = [
        "demo",
        "demo/examples",
        "demo/examples/positive", 
        "demo/examples/negative"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print("✅ All required directories exist")
    
    # Check files
    required_files = [
        "demo/select_examples.py",
        "demo/gui.py",
        "demo/util.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required files exist")
    
    # Check demo examples
    demo_files = list_demo_files()
    total_examples = len(demo_files['positive']) + len(demo_files['negative'])
    
    print(f"\nDemo Examples:")
    print(f"  Positive: {len(demo_files['positive'])} files")
    print(f"  Negative: {len(demo_files['negative'])} files")
    print(f"  Total: {total_examples} files")
    
    if total_examples == 0:
        print("⚠️  No demo examples found. Run select_examples.py first.")
        return False
    else:
        print("✅ Demo examples found")
    
    return True


def generate_stl_thumbnail(stl_path: str, output_path: str, size: tuple = (200, 200)) -> bool:
    """
    Generate a thumbnail image for an STL file.
    
    Args:
        stl_path: Path to the STL file
        output_path: Path to save the thumbnail image
        size: Thumbnail dimensions (width, height)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import trimesh
        from PIL import Image, ImageDraw
        
        # Load the STL mesh
        mesh = trimesh.load_mesh(stl_path)
        
        # Create a simple 2D projection thumbnail as fallback
        fig, ax = plt.subplots(1, 1, figsize=(size[0]/100, size[1]/100), dpi=100)
        
        # Project vertices to 2D (simple orthographic projection)
        vertices = mesh.vertices
        
        # Project along Z-axis (top view)
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        # Create a scatter plot of the vertices
        ax.scatter(x_coords, y_coords, s=0.1, alpha=0.6, color='blue')
        
        # Set equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Error generating thumbnail for {stl_path}: {e}")
        # Create placeholder thumbnail
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', size, color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=(100, 100, 100), width=2)
            
            # Add filename text
            filename = os.path.basename(stl_path)
            display_name = filename[:15] + "..." if len(filename) > 15 else filename
            
            # Simple text positioning
            text_x, text_y = 10, size[1] // 2
            draw.text((text_x, text_y), display_name, fill=(50, 50, 50))
            
            img.save(output_path)
            return True
        except:
            return False

def ensure_thumbnails_exist(examples_dir: str) -> dict:
    """
    Ensure thumbnails exist for all STL files in the examples directory.
    
    Args:
        examples_dir: Directory containing positive/negative subdirectories
        
    Returns:
        Dictionary mapping file paths to thumbnail generation success
    """
    results = {}
    thumbnail_dir = os.path.join(examples_dir, 'thumbnails')
    
    for class_type in ['positive', 'negative']:
        class_dir = os.path.join(examples_dir, class_type)
        thumb_class_dir = os.path.join(thumbnail_dir, class_type)
        
        if not os.path.exists(class_dir):
            continue
            
        # Create thumbnail directory if it doesn't exist
        os.makedirs(thumb_class_dir, exist_ok=True)
        
        # Process each STL file
        import glob
        stl_files = []
        for ext in ['*.stl', '*.STL']:
            stl_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        for stl_path in stl_files:
            filename = os.path.basename(stl_path)
            thumb_path = os.path.join(thumb_class_dir, f"{filename}.png")
            
            # Generate thumbnail if it doesn't exist
            if not os.path.exists(thumb_path):
                print(f"Generating thumbnail for {filename}...")
                success = generate_stl_thumbnail(stl_path, thumb_path)
                results[stl_path] = success
            else:
                results[stl_path] = True
    
    return results

def create_weight_visualization_data(layer_activations: dict, max_points: int = 100) -> dict:
    """
    Create visualization-friendly weight data for frontend animation.
    
    Args:
        layer_activations: Raw layer activation data
        max_points: Maximum number of data points per layer
        
    Returns:
        Structured data for frontend visualization
    """
    viz_data = {
        'layers': [],
        'animation_frames': []
    }
    
    # Process each layer
    for layer_id, layer_data in layer_activations.items():
        if isinstance(layer_data, dict) and 'values' in layer_data:
            values = layer_data['values']
            layer_type = layer_data.get('type', 'other')
            
            # Sample values if too many
            if len(values) > max_points:
                indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
                sampled_values = [values[i] for i in indices]
            else:
                sampled_values = values
            
            # Normalize values for visualization
            if sampled_values:
                min_val = min(sampled_values)
                max_val = max(sampled_values)
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in sampled_values]
                else:
                    normalized = [0.5] * len(sampled_values)
            else:
                normalized = []
            
            # Create layer visualization data
            layer_viz = {
                'id': layer_id,
                'name': layer_data.get('name', layer_id),
                'type': layer_type,
                'values': normalized,
                'original_values': sampled_values,
                'node_count': len(sampled_values),
                'activation_strength': np.mean(normalized) if normalized else 0
            }
            
            viz_data['layers'].append(layer_viz)
    
    # Create animation frames (simulate "thinking" process)
    num_frames = 30
    for frame in range(num_frames):
        frame_data = {
            'frame': frame,
            'progress': frame / (num_frames - 1),
            'layer_states': []
        }
        
        for layer in viz_data['layers']:
            # Simulate activation building up over time
            progress = frame / (num_frames - 1)
            
            # Create animated activation pattern
            animated_values = []
            for i, val in enumerate(layer['values']):
                # Add some temporal dynamics
                phase = (i / len(layer['values'])) * 2 * np.pi
                time_factor = progress + 0.1 * np.sin(phase + progress * 4 * np.pi)
                animated_val = val * min(1.0, time_factor)
                animated_values.append(max(0, animated_val))
            
            frame_data['layer_states'].append({
                'layer_id': layer['id'],
                'values': animated_values,
                'overall_activation': np.mean(animated_values) if animated_values else 0
            })
        
        viz_data['animation_frames'].append(frame_data)
    
    return viz_data

if __name__ == "__main__":
    # Test utility functions
    validate_demo_setup()
    
    # Test thumbnail generation if examples exist
    examples_dir = "demo/examples"
    if os.path.exists(examples_dir):
        print("\nGenerating thumbnails...")
        results = ensure_thumbnails_exist(examples_dir)
        success_count = sum(1 for success in results.values() if success)
        print(f"Thumbnail generation: {success_count}/{len(results)} successful")