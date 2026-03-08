"""
U-Net with Transfer Learning - ResNet50 Encoder
Uses pre-trained ImageNet weights for givinjg  better performance
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import numpy as np

class UNetTransferLearning:
    """
    U-Net with pre-trained ResNet50 encoder
    
    Benefits:
    - Starts with ImageNet knowledge
    - Faster convergence
    - Better accuracy (typically 5-10% improvement)
    - Less training data needed
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=4):
        """
        Initialize U-Net with transfer learning
        
        Args:
            input_shape: Input image dimensions
            num_classes: Number of segmentation classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        print(f"Transfer Learning U-Net initialized")
        print(f"   Using ResNet50 pre-trained encoder")
        print(f"   Input shape: {input_shape}")
        print(f"   Classes: {num_classes}")
    
    def build_model(self, freeze_encoder=True):
        """
        Build U-Net with ResNet50 encoder
        
        Args:
            freeze_encoder: If True, freeze encoder weights initially
                           (fine-tune later for best results)
        
        Architecture:
        Input (256x256x3)
            ↓
        ResNet50 Encoder (ImageNet weights)
            ├── Block 1 output (64x64x256)
            ├── Block 2 output (32x32x512)
            ├── Block 3 output (16x16x1024)
            └── Block 4 output (8x8x2048)
            ↓
        Custom Decoder (with skip connections)
            ↓
        Output (256x256x4)
        """
        
        print("\nBuilding Transfer Learning U-Net...")
        
        # ============================================================
        # ENCODER: ResNet50 (Pre-trained on ImageNet)
        # ============================================================
        
        print("   Loading ResNet50 encoder (ImageNet weights)...")
        
        # Load ResNet50 without top (classification) layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Extract intermediate layers for skip connections
        # These correspond to different resolution levels
        skip_connection_layers = [
            'conv1_relu',        # 128x128x64
            'conv2_block3_out',  # 64x64x256
            'conv3_block4_out',  # 32x32x512
            'conv4_block6_out',  # 16x16x1024
        ]
        
        # Create encoder with intermediate outputs
        encoder_outputs = [
            base_model.get_layer(name).output 
            for name in skip_connection_layers
        ]
        encoder_outputs.append(base_model.output)  # Bottleneck: 8x8x2048
        
        encoder = keras.Model(
            inputs=base_model.input,
            outputs=encoder_outputs,
            name='resnet50_encoder'
        )
        
        # Optionally freeze encoder weights
        if freeze_encoder:
            encoder.trainable = False
            print("   Encoder frozen (will fine-tune later)")
        else:
            encoder.trainable = True
            print("   Encoder trainable")
        
        # ============================================================
        # DECODER: Custom upsampling path
        # ============================================================
        
        print("   Building decoder...")
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Get encoder outputs
        skip1, skip2, skip3, skip4, bottleneck = encoder(inputs)
        
        # Decoder Block 1: 8x8 -> 16x16
        x = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
        x = layers.Concatenate()([x, skip4])  # Skip connection
        x = self._decoder_block(x, 512, name='decoder_block1')
        
        # Decoder Block 2: 16x16 -> 32x32
        x = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip3])  # Skip connection
        x = self._decoder_block(x, 256, name='decoder_block2')
        
        # Decoder Block 3: 32x32 -> 64x64
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip2])  # Skip connection
        x = self._decoder_block(x, 128, name='decoder_block3')
        
        # Decoder Block 4: 64x64 -> 128x128
        x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skip1])  # Skip connection
        x = self._decoder_block(x, 64, name='decoder_block4')
        
        # Final upsampling: 128x128 -> 256x256
        x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        
        # Output layer
        outputs = layers.Conv2D(
            self.num_classes, 
            (1, 1), 
            activation='softmax',
            name='output'
        )(x)
        
        # Create final model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_ResNet50')
        
        print("Transfer Learning U-Net built successfully!")
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([
            keras.backend.count_params(w) 
            for w in self.model.trainable_weights
        ])
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")
        
        return self.model
    
    def _decoder_block(self, inputs, filters, name):
        """
        Decoder convolutional block
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            name: Block name
            
        Returns:
            Output tensor
        """
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv1')(inputs)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile model with optimizer and loss
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            print("Build model first using build_model()")
            return
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.MeanIoU(num_classes=self.num_classes)
            ]
        )
        
        print("Model compiled successfully!")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Sparse Categorical Crossentropy")
        print(f"   Metrics: Accuracy, Mean IoU")
    
    def unfreeze_encoder(self, learning_rate=0.00001):
        """
        Unfreeze encoder for fine-tuning
        
        Call this after initial training to fine-tune the entire model
        
        Args:
            learning_rate: Lower learning rate for fine-tuning
        """
        if self.model is None:
            print("No model to unfreeze")
            return
        
        # Find encoder layers
        for layer in self.model.layers:
            if 'resnet50_encoder' in layer.name:
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=learning_rate)
        
        trainable_params = sum([
            keras.backend.count_params(w) 
            for w in self.model.trainable_weights
        ])
        
        print("Encoder unfrozen for fine-tuning!")
        print(f"   New learning rate: {learning_rate}")
        print(f"   Trainable parameters: {trainable_params:,}")
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            print("Build model first")
            return
        
        self.model.summary()
    
    def get_model(self):
        """Return the Keras model"""
        return self.model
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")