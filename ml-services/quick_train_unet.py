"""
Quick U-Net Training Script
Trains U-Net model with minimal epochs for demonstration
"""

import sys
sys.path.append('models')

from train_unet import UNetTrainer
from data_generator import create_synthetic_dataset

print("="*70)
print("⚡ QUICK U-NET TRAINING (Demo Mode)")
print("="*70)
print("\nThis will train a U-Net model on synthetic data")
print("For demonstration purposes only - uses minimal epochs")
print("\n" + "="*70)

# Create synthetic data
print("\n Creating synthetic dataset...")
data_dir = create_synthetic_dataset(num_samples=50, output_dir='data/synthetic')

# Initialize trainer
trainer = UNetTrainer()

# Prepare data
train_gen, val_gen = trainer.prepare_data(data_dir, train_split=0.8)

# Build model
trainer.build_and_compile(learning_rate=0.001)

# Quick training (just 10 epochs for demo)
print("\n🚀 Starting quick training (10 epochs)...")
print("   For real training, use 50-100 epochs")

history = trainer.train(train_gen, val_gen, epochs=10)

# Plot results
trainer.plot_training_history(save_path='outputs/training_curves.png')

# Evaluate
trainer.evaluate(val_gen)

print("\n" + "="*70)
print("QUICK TRAINING COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("   models/saved_models/unet_best.h5")
print("   models/saved_models/unet_final.h5")
print("   outputs/training_curves.png")
print("\nNext steps:")
print("  1. Test prediction: python models/predict_unet.py")
print("  2. Use in API: POST /api/segment")
print("="*70)