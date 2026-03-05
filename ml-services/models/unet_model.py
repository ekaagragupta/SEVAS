"""
U-Net Model for SEVAS
Semantic segmentation for environmental violation detection

U-Net Architecture:
- Encoder (Downsampling): Extract features
- Bottleneck: Deepest representation
- Decoder (Upsampling): Reconstruct segmentation mask
- Skip Connections: Preserve spatial information
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class UNetModel:
    """
    U-Net architecture for pixel-wise classification
    
    Input: RGB image (256, 256, 3)
    Output: Segmentation mask (256, 256, num_classes)
    
    Classes:
    0 - Background/Normal land
    1 - Sand mining/Disturbance
    2 - Healthy vegetation
    3 - Water bodies
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=4):
        """
        Initialize U-Net model
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of segmentation classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        print(f"✅ U-Net Model initialized")
        print(f"   Input shape: {input_shape}")
        print(f"   Classes: {num_classes}")
    
    def build_model(self):
        """
        Build complete U-Net architecture
        
        Architecture:
        
        Encoder (Contracting Path):
        Input (256x256) → Conv → Conv → MaxPool → (128x128)
                                    ↓
        (128x128) → Conv → Conv → MaxPool → (64x64)
                                    ↓
        (64x64) → Conv → Conv → MaxPool → (32x32)
                                    ↓
        Bottleneck (16x16)
                                    ↓
        Decoder (Expanding Path):
        (16x16) → UpConv → Concat(skip) → Conv → (32x32)
                                            ↓
        (32x32) → UpConv → Concat(skip) → Conv → (64x64)
                                            ↓
        (64x64) → UpConv → Concat(skip) → Conv → (128x128)
                                            ↓
        (128x128) → UpConv → Concat(skip) → Conv → (256x256)
                                            ↓
        Output (256x256x4)
        """
        
        print("\n🏗️  Building U-Net architecture...")
        
        inputs = keras.Input(shape=self.input_shape)
        
        # ==========================================
        # ENCODER (Contracting Path)
        # ==========================================
        
        # Block 1
        conv1 = self._conv_block(inputs, 64, name="encoder_block1")
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = self._conv_block(pool1, 128, name="encoder_block2")
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = self._conv_block(pool2, 256, name="encoder_block3")
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Block 4
        conv4 = self._conv_block(pool3, 512, name="encoder_block4")
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # ==========================================
        # BOTTLENECK
        # ==========================================
        
        bottleneck = self._conv_block(pool4, 1024, name="bottleneck")
        
        # ==========================================
        # DECODER (Expanding Path)
        # ==========================================
        
        # Block 5 (Upsample + Skip Connection)
        up5 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
        concat5 = layers.Concatenate()([up5, conv4])  # Skip connection
        conv5 = self._conv_block(concat5, 512, name="decoder_block5")
        
        # Block 6
        up6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
        concat6 = layers.Concatenate()([up6, conv3])  # Skip connection
        conv6 = self._conv_block(concat6, 256, name="decoder_block6")
        
        # Block 7
        up7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
        concat7 = layers.Concatenate()([up7, conv2])  # Skip connection
        conv7 = self._conv_block(concat7, 128, name="decoder_block7")
        
        # Block 8
        up8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
        concat8 = layers.Concatenate()([up8, conv1])  # Skip connection
        conv8 = self._conv_block(concat8, 64, name="decoder_block8")
        
        # ==========================================
        # OUTPUT LAYER
        # ==========================================
        
        # Final 1x1 convolution to get class probabilities
        outputs = layers.Conv2D(
            self.num_classes, 
            (1, 1), 
            activation='softmax',
            name='output'
        )(conv8)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='UNet')
        
        print("✅ U-Net architecture built successfully!")
        print(f"   Total layers: {len(self.model.layers)}")
        
        return self.model
    
    def _conv_block(self, inputs, filters, name):
        """
        Convolutional block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
        
        This is the basic building block used throughout U-Net
        
        Args:
            inputs: Input tensor
            filters: Number of filters
            name: Block name
            
        Returns:
            Output tensor after two conv operations
        """
        # First convolution
        x = layers.Conv2D(
            filters, 
            (3, 3), 
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv1'
        )(inputs)
        x = layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = layers.Activation('relu', name=f'{name}_relu1')(x)
        
        # Second convolution
        x = layers.Conv2D(
            filters, 
            (3, 3), 
            padding='same',
            kernel_initializer='he_normal',
            name=f'{name}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = layers.Activation('relu', name=f'{name}_relu2')(x)
        
        return x
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            print("⚠️  Build model first using build_model()")
            return
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # For integer labels
            metrics=[
                'accuracy',
                keras.metrics.MeanIoU(num_classes=self.num_classes)
            ]
        )
        
        print("✅ Model compiled successfully!")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Sparse Categorical Crossentropy")
        print(f"   Metrics: Accuracy, Mean IoU")
    
    def summary(self):
        """Print model architecture summary"""
        if self.model is None:
            print("⚠️  Build model first using build_model()")
            return
        
        self.model.summary()
    
    def get_model(self):
        """Return the Keras model"""
        return self.model
    
    def save_model(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save model (e.g., 'models/unet_trained.h5')
        """
        if self.model is None:
            print("⚠️  No model to save")
            return
        
        self.model.save(filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"✅ Model loaded from: {filepath}")
    
    def predict(self, image):
        """
        Predict segmentation mask for an image
        
        Args:
            image: Input image (256, 256, 3) - normalized 0-1
            
        Returns:
            Segmentation mask (256, 256) with class labels
        """
        if self.model is None:
            print("⚠️  Build/load model first")
            return None
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        
        # Get class with highest probability for each pixel
        mask = np.argmax(prediction[0], axis=-1)
        
        return mask
    
    def predict_with_confidence(self, image):
        """
        Predict with confidence scores
        
        Args:
            image: Input image
            
        Returns:
            mask: Predicted class for each pixel
            confidence: Confidence score for each pixel
        """
        if self.model is None:
            return None, None
        
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get probabilities
        prediction = self.model.predict(image, verbose=0)[0]
        
        # Get predicted class and confidence
        mask = np.argmax(prediction, axis=-1)
        confidence = np.max(prediction, axis=-1)
        
        return mask, confidence