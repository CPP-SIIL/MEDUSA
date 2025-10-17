"""
Main Training Script for STL Classification with Graph Neural Networks
Trains a GNN model to classify STL 3D models as gun parts or not.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import time

from gnn_model import STLClassifier
from data_loader import create_data_loaders, analyze_dataset


class Trainer:
    """Training class for STL classification model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.00001,
                 weight_decay: float = 1e-4,
                 class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.1,
                 gradient_clip_val: float = 1.0,
                 use_cosine_schedule: bool = True):
        """
        Initialize the trainer.
        
        Args:
            model: The GNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to run training on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            class_weights: Class weights for handling imbalanced data
            label_smoothing: Label smoothing factor
            gradient_clip_val: Gradient clipping value
            use_cosine_schedule: Whether to use cosine annealing scheduler
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        
        # Enhanced loss function with label smoothing
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
        # Enhanced optimizer with different parameters for different layers
        param_groups = [
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': learning_rate * 2},
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': learning_rate}
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # Enhanced learning rate scheduler
        if use_cosine_schedule:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.7, patience=8, min_lr=learning_rate * 0.01
            )
        
        self.use_cosine_schedule = use_cosine_schedule
        
        # Model averaging for better generalization
        self.model_ema = None
        self.ema_decay = 0.999
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        # TensorBoard writer
        self.writer = SummaryWriter(f'runs/stl_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    def update_ema_model(self):
        """Update exponential moving average model."""
        if self.model_ema is None:
            self.model_ema = type(self.model)(
                input_dim=self.model.input_dim if hasattr(self.model, 'input_dim') else 9,
                hidden_dim=self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else 64
            ).to(self.device)
            self.model_ema.load_state_dict(self.model.state_dict())
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (batch_graph, batch_labels) in enumerate(progress_bar):
            batch_graph = batch_graph.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_graph)
                    loss = self.criterion(outputs, batch_labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch_graph)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
            
            # Update EMA model
            self.update_ema_model()
            
            # Update learning rate for cosine schedule
            if self.use_cosine_schedule:
                self.scheduler.step()
            
            # Calculate gradient norm for monitoring
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Log to TensorBoard every 10 batches
            if batch_idx % 10 == 0:
                global_step = len(self.train_losses) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/Loss', batch_loss, global_step)
                self.writer.add_scalar('Batch/Accuracy', 100 * correct / total, global_step)
                self.writer.add_scalar('Batch/Gradient_Norm', total_norm, global_step)
                self.writer.add_scalar('Batch/Learning_Rate', self.optimizer.param_groups[0]['lr'], global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{100 * correct / total:.2f}%',
                'Grad_Norm': f'{total_norm:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> tuple[float, float, dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_ema_predictions = []
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for batch_graph, batch_labels in self.val_loader:
                batch_graph = batch_graph.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Regular model predictions
                outputs = self.model(batch_graph)
                loss = self.criterion(outputs, batch_labels)
                
                # EMA model predictions for comparison
                if self.model_ema is not None:
                    ema_outputs = self.model_ema(batch_graph)
                    _, ema_predicted = torch.max(ema_outputs, 1)
                    all_ema_predictions.extend(ema_predicted.cpu().numpy())
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Per-class accuracy
                for i in range(batch_labels.size(0)):
                    label = batch_labels[i].item()
                    class_correct[label] += (predicted[i] == batch_labels[i]).item()
                    class_total[label] += 1
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        # Calculate EMA model accuracy if available
        ema_accuracy = 0
        if self.model_ema is not None and all_ema_predictions:
            ema_correct = sum(1 for p, l in zip(all_ema_predictions, all_labels) if p == l)
            ema_accuracy = 100 * ema_correct / len(all_labels)
        
        # Calculate per-class accuracies
        class_accuracies = []
        for i in range(2):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'ema_accuracy': ema_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_0_accuracy': class_accuracies[0],
            'class_1_accuracy': class_accuracies[1],
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, accuracy, metrics

    def train(self, num_epochs: int = 100, patience: int = 20) -> dict:
        """Train the model."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Log model architecture to TensorBoard
        self.writer.add_text('Model/Architecture', str(self.model))
        self.writer.add_text('Model/Parameters', f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch()
            
            # Learning rate scheduling for ReduceLROnPlateau
            if not self.use_cosine_schedule:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
            else:
                old_lr = new_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start_time
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Comprehensive TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            if val_metrics['ema_accuracy'] > 0:
                self.writer.add_scalar('Accuracy/EMA_Validation', val_metrics['ema_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val_Class_0', val_metrics['class_0_accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val_Class_1', val_metrics['class_1_accuracy'], epoch)
            self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('Metrics/F1_Score', val_metrics['f1'], epoch)
            self.writer.add_scalar('Training/Learning_Rate', new_lr, epoch)
            self.writer.add_scalar('Training/Epoch_Time', epoch_time, epoch)
            
            # Log parameter histograms every 5 epochs
            if epoch % 5 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
            
            # Log confusion matrix every 10 epochs
            if epoch % 10 == 0:
                cm = confusion_matrix(val_metrics['labels'], val_metrics['predictions'])
                # Create confusion matrix plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - Epoch {epoch + 1}')
                self.writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
                plt.close(fig)
            
            # Log learning rate changes
            if old_lr != new_lr:
                self.writer.add_text('Training/LR_Change', 
                                   f'Epoch {epoch + 1}: LR changed from {old_lr:.6f} to {new_lr:.6f}')
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_metrics['ema_accuracy'] > 0:
                print(f"EMA Val Acc: {val_metrics['ema_accuracy']:.2f}%")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Class 0 Acc: {val_metrics['class_0_accuracy']:.2f}%, Class 1 Acc: {val_metrics['class_1_accuracy']:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s, LR: {new_lr:.6f}")
            
            # Use EMA model for best model selection if available
            comparison_acc = val_metrics['ema_accuracy'] if val_metrics['ema_accuracy'] > 0 else val_acc
            
            # Save best model (using validation accuracy instead of loss for better performance)
            if comparison_acc > getattr(self, 'best_val_acc', 0):
                self.best_val_acc = comparison_acc
                self.best_val_loss = val_loss
                if self.model_ema is not None and val_metrics['ema_accuracy'] > val_acc:
                    self.best_model_state = self.model_ema.state_dict().copy()
                else:
                    self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
                print(f"New best model saved! Val Acc: {comparison_acc:.2f}%")
                self.writer.add_text('Training/Best_Model', f'New best model at epoch {epoch + 1} with val_acc: {comparison_acc:.4f}')
            else:
                patience_counter += 1
            
            # Log early stopping progress
            self.writer.add_scalar('Training/Patience_Counter', patience_counter, epoch)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.writer.add_text('Training/Early_Stop', f'Training stopped at epoch {epoch + 1} due to no improvement for {patience} epochs')
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {best_epoch + 1}")
        
        # Log final training summary
        self.writer.add_text('Training/Summary', 
                           f'Training completed. Best epoch: {best_epoch + 1}, Best val loss: {self.best_val_loss:.4f}')
        
        self.writer.close()
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_acc': self.train_accuracies[-1],
            'final_val_acc': self.val_accuracies[-1]
        }
    
    def test(self) -> dict:
        """Test the model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_graph, batch_labels in tqdm(self.test_loader, desc="Testing"):
                batch_graph = batch_graph.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_graph)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:")
        print(cm)
        
        return test_metrics
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    data_dir = "dataset"
    output_dir = "outputs"
    batch_size = 12  # Reduced batch size for better gradient updates
    epochs = 200
    lr = 2e-4  # Slightly higher learning rate
    hidden_dim = 128  # Increased hidden dimension
    max_vertices = 500
    max_edges = 2000
    patience = 25  # Increased patience
    model_type = "full"  # Use full model for better capacity
    seed = 4738
    label_smoothing = 0.1
    gradient_clip_val = 1.0
    use_cosine_schedule = True

    args = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "max_vertices": max_vertices,
        "max_edges": max_edges,
        "patience": patience,
        "model_type": model_type,
        "seed": seed,
        "label_smoothing": label_smoothing,
        "gradient_clip_val": gradient_clip_val,
        "use_cosine_schedule": use_cosine_schedule
    }

    torch.manual_seed(seed)

    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Analyze dataset
    print("Analyzing dataset...")
    dataset_stats = analyze_dataset(data_dir)
    print(f"Dataset statistics: {dataset_stats}")

    for key, value in dataset_stats.items():
        if isinstance(value, np.int64):
            dataset_stats[key] = int(value)
        elif isinstance(value, np.float64):
             dataset_stats[key] = float(value)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        batch_size=batch_size,
        max_vertices=max_vertices,
        max_edges=max_edges
    )
    
    # Get class weights for imbalanced dataset
    class_weights = None
    if dataset_stats['class_balance'] < 0.3 or dataset_stats['class_balance'] > 0.7:
        print("Dataset is imbalanced, using class weights")
        # Calculate class weights
        pos_weight = dataset_stats['negative_samples'] / dataset_stats['positive_samples']
        class_weights = torch.tensor([1.0, pos_weight])
    
    # Create model
    print("Creating model...")
    model = STLClassifier(
        input_dim=9,  # Node feature dimension from STL processor
        hidden_dim=hidden_dim
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=lr,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        gradient_clip_val=gradient_clip_val,
        use_cosine_schedule=use_cosine_schedule
    )
    
    # Log initial dataset and model info to TensorBoard
    trainer.writer.add_text('Dataset/Statistics', str(dataset_stats))
    trainer.writer.add_text('Training/Configuration', str(args))
    
    # Train model
    print("Starting training...")
    training_results = trainer.train(num_epochs=epochs, patience=patience)
    
    # Test model
    print("Testing model...")
    test_results = trainer.test()
    
    full_output_path = os.path.join(output_dir, str(len(os.listdir(output_dir))) + "_" + str(time.time()))
    os.makedirs(full_output_path, exist_ok=True)

    # Plot training history
    trainer.plot_training_history(save_path=os.path.join(full_output_path, 'training_history.png'))
    
    # Save results
    results = {
        'args': args,
        'dataset_stats': dataset_stats,
        'training_results': training_results,
        'test_results': test_results
    }
    
    with open(os.path.join(full_output_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model_type,
            'input_dim': 9,
            'hidden_dim': hidden_dim
        },
        'results': results
    }, os.path.join(full_output_path, 'model.pt'))
    
    print(f"Training completed! Results saved to {full_output_path}")


if __name__ == "__main__":
    main()