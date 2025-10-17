"""
Enhanced Select Examples Script for Demo
Loads a trained model and identifies correctly classified training examples,
then copies them to demo directories for demonstration purposes.

KEY FEATURES:
✓ Confidence-based sorting: Automatically selects HIGHEST confidence examples
✓ Detailed analysis: Shows confidence distribution and statistics  
✓ Smart selection: Filters by minimum confidence threshold
✓ Comprehensive reporting: Detailed selection summary and JSON report
✓ No command line args: Simple variable-based configuration

CONFIGURATION:
- Edit the parameters in the main() function to customize selection
- Change model_folder to use a different trained model
- Adjust num_per_class and min_confidence as needed
- Set verbose and show_individual_files for detailed output
- No command line arguments required - just run: python demo/select_examples.py

SORTING BEHAVIOR:
- Finds all correctly classified examples
- Sorts by confidence score (highest first) 
- Applies minimum confidence filter
- Selects top N examples per class
- Shows confidence ranges and averages
"""

import os
import sys
import shutil
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_model import STLClassifier
from data_loader import STLDataset, collate_fn
from stl_processor import STLToGraphConverter


def list_available_models(outputs_dir: str = "outputs") -> List[str]:
    """List all available model folders with trained models."""
    if not os.path.exists(outputs_dir):
        return []
    
    model_folders = []
    for folder in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder)
        if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, 'model.pth')):
            model_folders.append(folder)
    
    return sorted(model_folders)


def get_latest_model(outputs_dir: str = "outputs") -> str:
    """Get the most recently modified model folder."""
    model_folders = list_available_models(outputs_dir)
    if not model_folders:
        return ""
    
    latest_folder = ""
    latest_time = 0
    
    for folder in model_folders:
        folder_path = os.path.join(outputs_dir, folder)
        model_path = os.path.join(folder_path, 'model.pth')
        mod_time = os.path.getmtime(model_path)
        if mod_time > latest_time:
            latest_time = mod_time
            latest_folder = folder
    
    return latest_folder


class ExampleSelector:
    """Class to select correctly classified examples for demo."""
    
    def __init__(self, model_folder: str, data_dir: str = "dataset"):
        """
        Initialize the example selector.
        
        Args:
            model_folder: Path to the trained model folder in outputs/
            data_dir: Path to the dataset directory
        """
        self.model_folder = model_folder
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and configuration
        self.model, self.model_config = self._load_model()
        self.model.eval()
        
        # Load dataset
        self.dataset = STLDataset(
            data_dir,
            max_vertices=500,  # Use default values from training
            max_edges=2000
        )
        
        print(f"Loaded model from {model_folder}")
        print(f"Model type: {self.model_config.get('model_type', 'unknown')}")
        print(f"Dataset: {len(self.dataset)} samples")
    
    def _load_model(self) -> Tuple[torch.nn.Module, Dict]:
        """Load the trained model from the model folder."""
        model_path = os.path.join(self.model_folder, 'model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint.get('model_config', {})
        
        # Create model instance
        model = STLClassifier(
            input_dim=model_config.get('input_dim', 9),
            hidden_dim=model_config.get('hidden_dim', 64)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model, model_config
    
    def _build_file_index(self):
        """Build an index mapping graph indices to file paths."""
        if hasattr(self, '_file_index'):
            return self._file_index
            
        print("Building file index...")
        
        # Get file paths from dataset
        positive_dir = os.path.join(self.data_dir, 'positive')
        negative_dir = os.path.join(self.data_dir, 'negative')
        
        file_paths = []
        labels = []
        
        # Collect positive files
        if os.path.exists(positive_dir):
            positive_files = glob.glob(os.path.join(positive_dir, '*.stl')) + \
                           glob.glob(os.path.join(positive_dir, '*.STL'))
            file_paths.extend(positive_files)
            labels.extend([1] * len(positive_files))
        
        # Collect negative files
        if os.path.exists(negative_dir):
            negative_files = glob.glob(os.path.join(negative_dir, '*.stl')) + \
                           glob.glob(os.path.join(negative_dir, '*.STL'))
            file_paths.extend(negative_files)
            labels.extend([0] * len(negative_files))
        
        # Build index by matching with dataset labels
        self._file_index = {}
        dataset_labels = self.dataset.labels
        
        # Simple matching based on order and labels
        pos_files = [f for f, l in zip(file_paths, labels) if l == 1]
        neg_files = [f for f, l in zip(file_paths, labels) if l == 0]
        
        pos_idx = 0
        neg_idx = 0
        
        for i, dataset_label in enumerate(dataset_labels):
            if dataset_label == 1 and pos_idx < len(pos_files):
                self._file_index[i] = pos_files[pos_idx]
                pos_idx += 1
            elif dataset_label == 0 and neg_idx < len(neg_files):
                self._file_index[i] = neg_files[neg_idx]
                neg_idx += 1
        
        print(f"Built file index for {len(self._file_index)} graphs")
        return self._file_index
    
    def _get_original_file_path(self, graph_idx: int) -> str:
        """Get the original STL file path for a graph index."""
        file_index = self._build_file_index()
        
        if graph_idx in file_index:
            return file_index[graph_idx]
        else:
            raise IndexError(f"Graph index {graph_idx} not found in file index")
    
    def evaluate_model(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Evaluate the model on the dataset and return predictions.
        
        Returns:
            Tuple of (predictions, true_labels, confidence_scores)
        """
        predictions = []
        true_labels = []
        confidence_scores = []
        
        print("Evaluating model on dataset...")
        
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset)), desc="Processing samples"):
                graph, label = self.dataset[i]
                graph = graph.to(self.device)
                
                # Create a batch with single graph
                from torch_geometric.data import Batch
                batch_graph = Batch.from_data_list([graph])
                
                # Make prediction
                output = self.model(batch_graph)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
                
                predictions.append(predicted_class)
                true_labels.append(label)
                confidence_scores.append(confidence)
        
        return predictions, true_labels, confidence_scores
    
    def analyze_confidence_distribution(self, predictions: List[int], true_labels: List[int], 
                                      confidence_scores: List[float]) -> None:
        """Analyze and display confidence distribution of predictions."""
        correct_mask = [p == t for p, t in zip(predictions, true_labels)]
        correct_confidences = [c for c, correct in zip(confidence_scores, correct_mask) if correct]
        incorrect_confidences = [c for c, correct in zip(confidence_scores, correct_mask) if not correct]
        
        # Overall statistics
        print(f"\nConfidence Analysis:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Correct predictions: {sum(correct_mask)} ({100*sum(correct_mask)/len(predictions):.1f}%)")
        
        if correct_confidences:
            print(f"  Correct predictions confidence: {min(correct_confidences):.3f} - {max(correct_confidences):.3f} (avg: {np.mean(correct_confidences):.3f})")
        
        # Per-class correct predictions
        pos_correct = [c for i, (c, correct, label) in enumerate(zip(confidence_scores, correct_mask, true_labels)) 
                      if correct and label == 1]
        neg_correct = [c for i, (c, correct, label) in enumerate(zip(confidence_scores, correct_mask, true_labels)) 
                      if correct and label == 0]
        
        if pos_correct:
            print(f"  Positive correct confidence: {min(pos_correct):.3f} - {max(pos_correct):.3f} (avg: {np.mean(pos_correct):.3f}, count: {len(pos_correct)})")
        if neg_correct:
            print(f"  Negative correct confidence: {min(neg_correct):.3f} - {max(neg_correct):.3f} (avg: {np.mean(neg_correct):.3f}, count: {len(neg_correct)})")
    
    def select_examples(self, 
                       num_per_class: int = 10,
                       min_confidence: float = 0.8) -> Dict[str, List[Dict]]:
        """
        Select correctly classified examples with high confidence.
        
        Args:
            num_per_class: Number of examples to select per class
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with selected examples for each class
        """
        predictions, true_labels, confidence_scores = self.evaluate_model()
        
        # Analyze confidence distribution
        self.analyze_confidence_distribution(predictions, true_labels, confidence_scores)
        
        # Find ALL correctly classified examples (before confidence filtering)
        all_correct_indices = []
        for i, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidence_scores)):
            if pred == true:
                all_correct_indices.append({
                    'index': i,
                    'label': true,
                    'confidence': conf,
                    'predicted': pred
                })
        
        print(f"\nFound {len(all_correct_indices)} total correctly classified examples")
        
        # Show top examples before filtering
        all_pos = sorted([ex for ex in all_correct_indices if ex['label'] == 1], 
                        key=lambda x: x['confidence'], reverse=True)
        all_neg = sorted([ex for ex in all_correct_indices if ex['label'] == 0], 
                        key=lambda x: x['confidence'], reverse=True)
        
        if all_pos:
            top_pos_conf = [f"{ex['confidence']:.3f}" for ex in all_pos[:5]]
            print(f"  Top positive confidences: {top_pos_conf}")
        if all_neg:
            top_neg_conf = [f"{ex['confidence']:.3f}" for ex in all_neg[:5]]
            print(f"  Top negative confidences: {top_neg_conf}")
        
        # Apply confidence threshold filter
        correct_indices = [ex for ex in all_correct_indices if ex['confidence'] >= min_confidence]
        print(f"After applying min_confidence >= {min_confidence}: {len(correct_indices)} examples remaining")
        
        # Separate by class and sort by confidence (highest first)
        positive_examples = [ex for ex in correct_indices if ex['label'] == 1]
        negative_examples = [ex for ex in correct_indices if ex['label'] == 0]
        
        print(f"  - Positive examples found: {len(positive_examples)}")
        print(f"  - Negative examples found: {len(negative_examples)}")
        
        # Sort by confidence in descending order (highest confidence first)
        positive_examples = sorted(positive_examples, key=lambda x: x['confidence'], reverse=True)
        negative_examples = sorted(negative_examples, key=lambda x: x['confidence'], reverse=True)
        
        # Select top N examples for each class (highest confidence)
        selected_positive = positive_examples[:min(num_per_class, len(positive_examples))]
        selected_negative = negative_examples[:min(num_per_class, len(negative_examples))]
        
        selected = {
            'positive': selected_positive,
            'negative': selected_negative
        }
        
        # Print selection summary with confidence ranges
        print(f"\nSelected TOP {len(selected_positive)} positive examples (highest confidence):")
        if selected_positive:
            confidences = [ex['confidence'] for ex in selected_positive]
            print(f"  - Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  - Average confidence: {np.mean(confidences):.3f}")
        
        print(f"Selected TOP {len(selected_negative)} negative examples (highest confidence):")
        if selected_negative:
            confidences = [ex['confidence'] for ex in selected_negative]
            print(f"  - Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  - Average confidence: {np.mean(confidences):.3f}")
        
        return selected
    
    def copy_examples(self, selected_examples: Dict[str, List[Dict]], 
                     demo_dir: str = "demo/examples", 
                     show_individual_files: bool = True) -> Dict[str, List[str]]:
        """
        Copy selected examples to demo directories.
        
        Args:
            selected_examples: Dictionary of selected examples
            demo_dir: Base demo directory
            
        Returns:
            Dictionary with copied file paths
        """
        copied_files = {'positive': [], 'negative': []}
        
        for class_name, examples in selected_examples.items():
            class_dir = os.path.join(demo_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"\nCopying {class_name} examples...")
            
            for i, example in enumerate(tqdm(examples, desc=f"Copying {class_name}")):
                try:
                    # Get original file path
                    original_path = self._get_original_file_path(example['index'])
                    
                    # Create new filename with confidence score
                    original_name = os.path.basename(original_path)
                    name_parts = os.path.splitext(original_name)
                    new_name = f"{name_parts[0]}_conf{example['confidence']:.3f}{name_parts[1]}"
                    
                    # Copy file
                    dest_path = os.path.join(class_dir, new_name)
                    shutil.copy2(original_path, dest_path)
                    copied_files[class_name].append(dest_path)
                    
                    if show_individual_files:
                        print(f"  Copied: {original_name} -> {new_name} (conf: {example['confidence']:.3f})")
                    
                except Exception as e:
                    print(f"  Error copying example {example['index']}: {e}")
        
        return copied_files
    
    def create_summary_report(self, selected_examples: Dict[str, List[Dict]], 
                            copied_files: Dict[str, List[str]], 
                            output_path: str = "demo/selection_report.json"):
        """Create a summary report of the selection process."""
        
        # Calculate overall accuracy
        predictions, true_labels, confidence_scores = self.evaluate_model()
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        
        # Create report
        report = {
            'model_info': {
                'model_folder': self.model_folder,
                'model_type': self.model_config.get('model_type', 'unknown'),
                'input_dim': self.model_config.get('input_dim', 9),
                'hidden_dim': self.model_config.get('hidden_dim', 64)
            },
            'dataset_info': {
                'total_samples': len(self.dataset),
                'positive_samples': sum(self.dataset.labels),
                'negative_samples': len(self.dataset.labels) - sum(self.dataset.labels)
            },
            'model_performance': {
                'overall_accuracy': accuracy,
                'total_correct': sum(1 for p, t in zip(predictions, true_labels) if p == t),
                'total_samples': len(predictions)
            },
            'selection_criteria': {
                'min_confidence': 0.8,  # Default value used
                'examples_per_class': len(selected_examples.get('positive', [])) + len(selected_examples.get('negative', []))
            },
            'selected_examples': {
                'positive': [
                    {
                        'filename': os.path.basename(path),
                        'confidence': example['confidence'],
                        'original_index': example['index']
                    }
                    for example, path in zip(selected_examples.get('positive', []), copied_files.get('positive', []))
                ],
                'negative': [
                    {
                        'filename': os.path.basename(path),
                        'confidence': example['confidence'],
                        'original_index': example['index']
                    }
                    for example, path in zip(selected_examples.get('negative', []), copied_files.get('negative', []))
                ]
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSelection report saved to: {output_path}")
        return report


def main():
    """
    Main function to run example selection.
    
    TO USE: Simply edit the configuration parameters below and run:
    python demo/select_examples.py
    
    No command line arguments needed!
    """
    
    # ===== CONFIGURATION PARAMETERS =====
    # Model configuration
    # Option 1: Specify exact model folder
    model_folder = "6_1759440938.2232356"  # Change this to your model folder name
    
    # Option 2: Use latest model (comment out model_folder above and uncomment below)
    # model_folder = get_latest_model()
    
    # Option 3: Choose from available models (comment out model_folder above and uncomment below)
    # available_models = list_available_models()
    # if available_models:
    #     print("Available models:")
    #     for i, model in enumerate(available_models):
    #         print(f"  {i}: {model}")
    #     choice = int(input("Select model index: "))
    #     model_folder = available_models[choice]
    # else:
    #     print("No models found!")
    #     return
    
    # Dataset configuration  
    data_dir = "dataset"
    
    # Selection parameters
    num_per_class = 20           # Number of examples to select per class
    min_confidence = 0.80       # Minimum confidence threshold (0.0 - 1.0)
    
    # Output configuration
    demo_dir = "demo/examples"  # Directory to save selected examples
    
    # Display options
    verbose = True              # Show detailed confidence analysis
    show_individual_files = True # Show individual file confidence scores
    
    # ===== END CONFIGURATION =====
    
    print("=== STL Classification Demo Example Selection ===")
    
    # Show available models for reference
    available_models = list_available_models()
    if available_models:
        print("Available trained models:")
        for i, model in enumerate(available_models):
            marker = " <- USING" if model == model_folder else ""
            print(f"  {i+1}. {model}{marker}")
        print()
    
    # Validate and construct full model path
    if not model_folder:
        print("Error: No model folder specified!")
        available_models = list_available_models()
        if available_models:
            print("Available model folders:")
            for model in available_models:
                print(f"  - {model}")
        return
    
    # Construct full model path
    if os.path.exists(model_folder):
        # Already a full path
        full_model_path = model_folder
    else:
        # Try as relative path from outputs/
        full_model_path = os.path.join('outputs', model_folder)
        if not os.path.exists(full_model_path):
            print(f"Error: Model folder not found: {model_folder}")
            available_models = list_available_models()
            if available_models:
                print("Available model folders:")
                for model in available_models:
                    print(f"  - {model}")
            return
    
    # Verify model.pth exists
    if not os.path.exists(os.path.join(full_model_path, 'model.pth')):
        print(f"Error: model.pth not found in {full_model_path}")
        return
    
    print(f"Model folder: {full_model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Examples per class: {num_per_class}")
    print(f"Minimum confidence: {min_confidence}")
    print(f"Demo directory: {demo_dir}")
    print()
    
    try:
        # Create example selector
        selector = ExampleSelector(full_model_path, data_dir)
        
        # Select examples
        print(f"\nSelecting examples with min confidence: {min_confidence}")
        selected_examples = selector.select_examples(
            num_per_class=num_per_class,
            min_confidence=min_confidence
        )
        
        # Copy examples to demo directory
        print(f"\nCopying examples to: {demo_dir}")
        copied_files = selector.copy_examples(selected_examples, demo_dir, show_individual_files)
        
        # Create summary report
        report = selector.create_summary_report(selected_examples, copied_files)
        
        # Print summary
        print(f"\n=== Selection Summary ===")
        print(f"Model accuracy: {report['model_performance']['overall_accuracy']:.3f}")
        print(f"Positive examples selected: {len(copied_files['positive'])}")
        print(f"Negative examples selected: {len(copied_files['negative'])}")
        print(f"Total examples copied: {len(copied_files['positive']) + len(copied_files['negative'])}")
        
        if copied_files['positive']:
            avg_conf_pos = np.mean([ex['confidence'] for ex in selected_examples['positive']])
            print(f"Average confidence (positive): {avg_conf_pos:.3f}")
        
        if copied_files['negative']:
            avg_conf_neg = np.mean([ex['confidence'] for ex in selected_examples['negative']])
            print(f"Average confidence (negative): {avg_conf_neg:.3f}")
        
        print(f"\nDemo files ready in: {demo_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()