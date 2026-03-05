"""
Data Generator for U-Net Training
Creates synthetic training data and handles augmentation
"""

import numpy as np
import cv2
from tensorflow import keras

class SegmentationDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for segmentation tasks
    
    Generates batches of images and corresponding masks
    Applies data augmentation on-the-fly
    """
    
    def __init__(self, image_paths, mask_paths, batch_size=8, 
                 image_size=(256, 256), num_classes=4, augment=True):
        """
        Initialize data generator
        
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to corresponding masks
            batch_size: Number of samples per batch
            image_size: Target image size
            num_classes: Number of segmentation classes
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        
        print(f" Data Generator initialized")
        print(f"   Samples: {len(self.image_paths)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Augmentation: {augment}")
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Returns:
            X: Batch of images (batch_size, height, width, 3)
            y: Batch of masks (batch_size, height, width)
        """
        # Get batch indexes
        batch_indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        
        # Initialize arrays
        X = np.zeros((len(batch_indexes), *self.image_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_indexes), *self.image_size), dtype=np.int32)
        
        # Load and preprocess each sample
        for i, idx in enumerate(batch_indexes):
            # Load image and mask
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Resize
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Apply augmentation
            if self.augment:
                image, mask = self._augment(image, mask)
            
            X[i] = image
            y[i] = mask
        
        return X, y
    
    def _augment(self, image, mask):
        """
        Apply random augmentation
        
        Augmentations:
        - Horizontal flip
        - Vertical flip
        - Rotation (90, 180, 270 degrees)
        - Brightness adjustment
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random rotation (90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        
        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        
        return image, mask
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        np.random.shuffle(self.indexes)


def create_synthetic_dataset(num_samples=100, output_dir='data/synthetic'):
    """
    Create synthetic training data for demonstration
    
    Since we don't have labeled satellite data, this creates
    simple synthetic examples for model training demonstration
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Where to save the data
    """
    import os
    
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/masks', exist_ok=True)
    
    print(f"\n🎨 Creating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        # Create synthetic image (256x256x3)
        image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        # Create synthetic mask (256x256) with 4 classes
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Class 1: Mining area (random circles)
        for _ in range(np.random.randint(1, 3)):
            cx, cy = np.random.randint(0, 256, 2)
            radius = np.random.randint(20, 50)
            cv2.circle(mask, (cx, cy), radius, 1, -1)
        
        # Class 2: Vegetation (random rectangles)
        for _ in range(np.random.randint(2, 4)):
            x1, y1 = np.random.randint(0, 200, 2)
            x2, y2 = x1 + np.random.randint(30, 80), y1 + np.random.randint(30, 80)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 2, -1)
        
        # Class 3: Water (random polygons)
        for _ in range(np.random.randint(1, 2)):
            points = np.random.randint(0, 256, (5, 2))
            cv2.fillPoly(mask, [points], 3)
        
        # Save
        cv2.imwrite(f'{output_dir}/images/sample_{i:04d}.png', image)
        cv2.imwrite(f'{output_dir}/masks/sample_{i:04d}.png', mask)
    
    print(f" Created {num_samples} synthetic samples in {output_dir}/")
    
    return output_dir