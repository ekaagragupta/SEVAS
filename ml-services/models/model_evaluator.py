"""
Model Evaluation Module for SEVAS
Comprehensive metrics and visualizations for model performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_fscore_support,
    jaccard_score
)
from tensorflow import keras
import cv2

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
        """
        self.model = keras.models.load_model(model_path)
        
        if class_names is None:
            self.class_names = ['Background', 'Mining', 'Vegetation', 'Water']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        
        print(f" Model Evaluator initialized")
        print(f"   Classes: {self.class_names}")
    
    def evaluate_dataset(self, X_val, y_val):
        """
        Comprehensive evaluation on validation set
        
        Args:
            X_val: Validation images (N, H, W, 3)
            y_val: Validation masks (N, H, W)
            
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "="*70)
        print(" COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Get predictions
        print("\n Generating predictions...")
        predictions = self.model.predict(X_val, verbose=1)
        y_pred = np.argmax(predictions, axis=-1)
        
        # Flatten arrays for sklearn metrics
        y_true_flat = y_val.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        results = {}
        
        # 1. Overall Accuracy
        print("\n Calculating metrics...")
        accuracy = np.mean(y_true_flat == y_pred_flat)
        results['overall_accuracy'] = accuracy
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 2. Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, 
            labels=range(self.num_classes),
            zero_division=0
        )
        
        results['per_class'] = {}
        print("\n Per-Class Metrics:")
        print("-"*70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
        print("-"*70)
        
        # 3. Mean IoU (Intersection over Union)
        iou_scores = []
        print("\n IoU (Intersection over Union) Scores:")
        print("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            # Calculate IoU for each class
            iou = jaccard_score(
                y_true_flat == i, 
                y_pred_flat == i,
                zero_division=0
            )
            iou_scores.append(iou)
            print(f"{class_name:<15} IoU: {iou:.4f}")
        
        mean_iou = np.mean(iou_scores)
        results['mean_iou'] = float(mean_iou)
        results['iou_per_class'] = {
            self.class_names[i]: float(iou) 
            for i, iou in enumerate(iou_scores)
        }
        print("-"*70)
        print(f"Mean IoU: {mean_iou:.4f}")
        
        # 4. Confusion Matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, 
                             labels=range(self.num_classes))
        results['confusion_matrix'] = cm.tolist()
        
        print("\n" + "="*70)
        print(" EVALUATION COMPLETE")
        print("="*70)
        
        return results, y_pred
    
    def plot_confusion_matrix(self, results, save_path='outputs/confusion_matrix.png'):
        """
        Plot confusion matrix heatmap
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Where to save the plot
        """
        cm = np.array(results['confusion_matrix'])
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Confusion matrix saved to: {save_path}")
    
    def plot_metrics_comparison(self, results, save_path='outputs/metrics_comparison.png'):
        """
        Plot comparison of precision, recall, F1-score
        
        Args:
            results: Results dictionary
            save_path: Where to save
        """
        classes = list(results['per_class'].keys())
        precision = [results['per_class'][c]['precision'] for c in classes]
        recall = [results['per_class'][c]['recall'] for c in classes]
        f1 = [results['per_class'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Metrics comparison saved to: {save_path}")
    
    def plot_iou_scores(self, results, save_path='outputs/iou_scores.png'):
        """
        Plot IoU scores per class
        
        Args:
            results: Results dictionary
            save_path: Where to save
        """
        classes = list(results['iou_per_class'].keys())
        iou_scores = list(results['iou_per_class'].values())
        mean_iou = results['mean_iou']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#9E9E9E', '#F44336', '#4CAF50', '#2196F3']
        bars = ax.bar(classes, iou_scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add mean IoU line
        ax.axhline(y=mean_iou, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean IoU: {mean_iou:.4f}')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
        ax.set_title('Intersection over Union (IoU) per Class', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, iou_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" IoU scores saved to: {save_path}")
    
    def plot_sample_predictions(self, X_val, y_val, y_pred, 
                                num_samples=4, save_path='outputs/sample_predictions.png'):
        """
        Plot sample predictions vs ground truth
        
        Args:
            X_val: Validation images
            y_val: True masks
            y_pred: Predicted masks
            num_samples: Number of samples to show
            save_path: Where to save
        """
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*3))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        colors = np.array([[0.8, 0.8, 0.8], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(X_val[i])
            axes[i, 0].set_title('Input Image', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_rgb = colors[y_val[i]]
            axes[i, 1].imshow(gt_rgb)
            axes[i, 1].set_title('Ground Truth', fontweight='bold')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_rgb = colors[y_pred[i]]
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title('Prediction', fontweight='bold')
            axes[i, 2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=self.class_names[i])
            for i in range(len(self.class_names))
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=4, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Sample predictions saved to: {save_path}")
    
    def generate_evaluation_report(self, results, save_path='outputs/evaluation_report.txt'):
        """
        Generate text report of evaluation
        
        Args:
            results: Results dictionary
            save_path: Where to save report
        """
        report = []
        report.append("="*70)
        report.append("SEVAS U-NET MODEL EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE")
        report.append("-"*70)
        report.append(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
        report.append(f"Mean IoU:         {results['mean_iou']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS")
        report.append("-"*70)
        report.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'IoU':<12}")
        report.append("-"*70)
        
        for class_name in self.class_names:
            metrics = results['per_class'][class_name]
            iou = results['iou_per_class'][class_name]
            report.append(
                f"{class_name:<15} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1_score']:<12.4f} "
                f"{iou:<12.4f}"
            )
        
        report.append("-"*70)
        report.append("")
        
        # Confusion matrix
        report.append("CONFUSION MATRIX")
        report.append("-"*70)
        cm = np.array(results['confusion_matrix'])
        
        # Header
        header = f"{'True \\ Pred':<15}"
        for class_name in self.class_names:
            header += f"{class_name:<12}"
        report.append(header)
        report.append("-"*70)
        
        # Rows
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name:<15}"
            for j in range(len(self.class_names)):
                row += f"{cm[i, j]:<12}"
            report.append(row)
        
        report.append("="*70)
        
        # Write to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Evaluation report saved to: {save_path}")
        
        # Also print to console
        print("\n" + '\n'.join(report))


def evaluate_model():
    """
    Main evaluation function
    """
    print("="*70)
    print("🔬 MODEL EVALUATION PIPELINE")
    print("="*70)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    
    # For demonstration, create small validation set
    # In production, use real validation data
    import os
    
    if os.path.exists('data/synthetic/images'):
        # Load synthetic data
        from data_generator import create_simple_data
        
        # Create simple validation set
        print("   Creating synthetic validation set...")
        X_val, y_val = create_simple_data(num_samples=20)
        
        # Split into actual val
        X_val = X_val[-10:]
        y_val = y_val[-10:]
        
        print(f"   Validation set: {X_val.shape}")
    else:
        print("⚠️  No validation data found")
        print("   Creating synthetic data...")
        
        # Create inline
        X_val = []
        y_val = []
        
        for i in range(10):
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            img = img.astype(np.float32) / 255.0
            
            mask = np.zeros((256, 256), dtype=np.int32)
            cv2.circle(mask, (128, 128), 30, 1, -1)
            cv2.rectangle(mask, (50, 50), (100, 100), 2, -1)
            
            X_val.append(img)
            y_val.append(mask)
        
        X_val = np.array(X_val)
        y_val = np.array(y_val)
    
    # Initialize evaluator
    print("\n🔧 Initializing evaluator...")
    evaluator = ModelEvaluator('models/saved_models/unet_best.h5')
    
    # Run evaluation
    results, y_pred = evaluator.evaluate_dataset(X_val, y_val)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    evaluator.plot_confusion_matrix(results)
    evaluator.plot_metrics_comparison(results)
    evaluator.plot_iou_scores(results)
    evaluator.plot_sample_predictions(X_val, y_val, y_pred, num_samples=4)
    
    # Generate report
    print("\n📝 Generating report...")
    evaluator.generate_evaluation_report(results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   ✅ outputs/confusion_matrix.png")
    print("   ✅ outputs/metrics_comparison.png")
    print("   ✅ outputs/iou_scores.png")
    print("   ✅ outputs/sample_predictions.png")
    print("   ✅ outputs/evaluation_report.txt")
    print("="*70)


if __name__ == '__main__':
    # Need this helper function
    def create_simple_data(num_samples):
        """Create simple synthetic data for evaluation"""
        import cv2
        X = []
        y = []
        
        for i in range(num_samples):
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            img = img.astype(np.float32) / 255.0
            
            mask = np.zeros((256, 256), dtype=np.int32)
            cv2.circle(mask, (np.random.randint(50, 200), np.random.randint(50, 200)), 
                      np.random.randint(20, 40), 1, -1)
            cv2.rectangle(mask, (np.random.randint(0, 150), np.random.randint(0, 150)),
                         (np.random.randint(100, 250), np.random.randint(100, 250)), 2, -1)
            
            X.append(img)
            y.append(mask)
        
        return np.array(X), np.array(y)
    
    evaluate_model()