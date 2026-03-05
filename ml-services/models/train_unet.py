"""
U-Net Training Script for SEVAS
Trains the segmentation model on satellite imagery
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from datetime import datetime

from unet_model import UNetModel
from data_generator import SegmentationDataGenerator, create_synthetic_dataset

class UNetTrainer:
    """
    Handles U-Net model training with callbacks and visualization
    """
    
    def __init__(self, model_dir='models/saved_models'):
        """
        Initialize trainer
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.unet = None
        self.history = None
        
        print("U-Net Trainer initialized")
        print(f"   Model directory: {model_dir}")
    
    def prepare_data(self, data_dir='data/synthetic', train_split=0.8):
        """
        Prepare training and validation data
        
        Args:
            data_dir: Directory containing images and masks
            train_split: Fraction of data for training
            
        Returns:
            train_gen: Training data generator
            val_gen: Validation data generator
        """
        print("\n Preparing data...")
        
        # Get all image and mask paths
        image_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        image_paths = [os.path.join(image_dir, f) for f in image_files]
        mask_paths = [os.path.join(mask_dir, f) for f in image_files]
        
        # Split into train and validation
        split_idx = int(len(image_paths) * train_split)
        
        train_images = image_paths[:split_idx]
        train_masks = mask_paths[:split_idx]
        val_images = image_paths[split_idx:]
        val_masks = mask_paths[split_idx:]
        
        print(f"   Total samples: {len(image_paths)}")
        print(f"   Training samples: {len(train_images)}")
        print(f"   Validation samples: {len(val_images)}")
        
        # Create data generators
        train_gen = SegmentationDataGenerator(
            train_images, train_masks,
            batch_size=8,
            augment=True
        )
        
        val_gen = SegmentationDataGenerator(
            val_images, val_masks,
            batch_size=8,
            augment=False
        )
        
        return train_gen, val_gen
    
    def build_and_compile(self, input_shape=(256, 256, 3), num_classes=4, learning_rate=0.0001):
        """
        Build and compile U-Net model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of segmentation classes
            learning_rate: Learning rate for optimizer
        """
        print("\n  Building U-Net model...")
        
        self.unet = UNetModel(input_shape=input_shape, num_classes=num_classes)
        self.unet.build_model()
        self.unet.compile_model(learning_rate=learning_rate)
        
        print("\n Model Architecture:")
        print("="*70)
        
        # Count parameters
        total_params = self.unet.model.count_params()
        print(f"Total Parameters: {total_params:,}")
        print(f"Estimated Size: {total_params * 4 / (1024**2):.2f} MB")
        print("="*70)
    
    def get_callbacks(self):
        """
        Create training callbacks
        
        Callbacks:
        - ModelCheckpoint: Save best model
        - EarlyStopping: Stop if no improvement
        - ReduceLROnPlateau: Reduce learning rate when stuck
        - TensorBoard: Logging for visualization
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'unet_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                os.path.join(self.model_dir, f'training_log_{timestamp}.csv')
            )
        ]
        
        return callbacks
    
    def train(self, train_gen, val_gen, epochs=50):
        """
        Train the U-Net model
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of training epochs
        """
        print("\n🚀 Starting training...")
        print("="*70)
        
        callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.unet.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n Training complete!")
        
        # Save final model
        final_path = os.path.join(self.model_dir, 'unet_final.h5')
        self.unet.save_model(final_path)
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("  No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Training plot saved to: {save_path}")
        
        plt.show()
    
    def evaluate(self, val_gen):
        """
        Evaluate model on validation set
        
        Args:
            val_gen: Validation data generator
            
        Returns:
            Evaluation metrics
        """
        if self.unet is None or self.unet.model is None:
            print(" No model to evaluate")
            return None
        
        print("\n Evaluating model...")
        
        results = self.unet.model.evaluate(val_gen, verbose=1)
        
        print("\n Evaluation Results:")
        print("="*70)
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print("="*70)
        
        return results


def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("🎓 U-NET TRAINING FOR SEVAS")
    print("="*70)
    
    # Step 1: Create synthetic dataset
    print("\nStep 1: Creating synthetic dataset...")
    data_dir = create_synthetic_dataset(num_samples=100)
    
    # Step 2: Initialize trainer
    print("\nStep 2: Initializing trainer...")
    trainer = UNetTrainer()
    
    # Step 3: Prepare data
    print("\nStep 3: Preparing data...")
    train_gen, val_gen = trainer.prepare_data(data_dir)
    
    # Step 4: Build model
    print("\nStep 4: Building model...")
    trainer.build_and_compile(
        input_shape=(256, 256, 3),
        num_classes=4,
        learning_rate=0.001
    )
    
    # Step 5: Train
    print("\nStep 5: Training model...")
    history = trainer.train(
        train_gen,
        val_gen,
        epochs=30  # Use 30 epochs for demo (increase for real training)
    )
    
    # Step 6: Plot results
    print("\nStep 6: Plotting results...")
    trainer.plot_training_history(save_path='outputs/training_history.png')
    
    # Step 7: Evaluate
    print("\nStep 7: Final evaluation...")
    trainer.evaluate(val_gen)
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Saved files:")
    print("   - models/saved_models/unet_best.h5 (best model)")
    print("   - models/saved_models/unet_final.h5 (final model)")
    print("   - outputs/training_history.png (training curves)")
    print("="*70)


if __name__ == '__main__':
    main()