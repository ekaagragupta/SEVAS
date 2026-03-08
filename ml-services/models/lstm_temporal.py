"""
LSTM Temporal Prediction for SEVAS
Predicts future environmental violations using time-series satellite data

Given: Sequence of 5-10 images over time (e.g., monthly)
Predicts: Probability of violation in next time period
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class LSTMTemporalPredictor:
    """
    LSTM-based temporal prediction for environmental monitoring
    
    Architecture:
    Input: Sequence of feature vectors from images
        ↓
    TimeDistributed CNN Feature Extractor
        ↓
    LSTM Layers (temporal patterns)
        ↓
    Dense Layers
        ↓
    Output: Violation probability + severity
    """
    
    def __init__(self, sequence_length=5, image_shape=(256, 256, 3)):
        """
        Initialize LSTM temporal predictor
        
        Args:
            sequence_length: Number of time steps (images)
            image_shape: Input image dimensions
        """
        self.sequence_length = sequence_length
        self.image_shape = image_shape
        self.model = None
        
        print(f"LSTM Temporal Predictor initialized")
        print(f"   Sequence length: {sequence_length} time steps")
        print(f"   Image shape: {image_shape}")
    
    def build_model(self):
        """
        Build LSTM temporal prediction model
        
        Architecture:
        1. CNN Feature Extractor (per image)
        2. LSTM for temporal patterns
        3. Dense layers for prediction
        """
        
        print("\nBuilding LSTM temporal model...")
        
        # Input: Sequence of images
        inputs = keras.Input(shape=(self.sequence_length, *self.image_shape))
        
        # ============================================================
        # CNN FEATURE EXTRACTOR (Applied to each time step)
        # ============================================================
        
        # TimeDistributed wrapper applies CNN to each image in sequence
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            name='td_conv1'
        )(inputs)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            name='td_conv2'
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            name='td_conv3'
        )(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            name='td_conv4'
        )(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        
        # Now we have: (sequence_length, 256) feature vectors
        
        # ============================================================
        # LSTM LAYERS (Temporal Pattern Learning)
        # ============================================================
        
        # First LSTM layer - returns sequences
        x = layers.LSTM(128, return_sequences=True, name='lstm1')(x)
        x = layers.Dropout(0.3)(x)
        
        # Second LSTM layer - returns final state
        x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)
        x = layers.Dropout(0.3)(x)
        
        # ============================================================
        # PREDICTION HEAD
        # ============================================================
        
        # Dense layers for final prediction
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Multi-output prediction
        # Output 1: Violation probability (binary)
        violation_prob = layers.Dense(1, activation='sigmoid', name='violation_probability')(x)
        
        # Output 2: Severity score (regression, 0-1)
        severity_score = layers.Dense(1, activation='sigmoid', name='severity_score')(x)
        
        # Output 3: Violation type (multi-class)
        violation_type = layers.Dense(4, activation='softmax', name='violation_type')(x)
        # Types: 0=None, 1=Mining, 2=Encroachment, 3=Vegetation Loss
        
        # Create model with multiple outputs
        self.model = keras.Model(
            inputs=inputs,
            outputs={
                'violation_probability': violation_prob,
                'severity_score': severity_score,
                'violation_type': violation_type
            },
            name='LSTM_Temporal_Predictor'
        )
        
        print("LSTM temporal model built!")
        
        total_params = self.model.count_params()
        print(f"   Total parameters: {total_params:,}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with multi-output losses
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            print("Build model first")
            return
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'violation_probability': 'binary_crossentropy',
                'severity_score': 'mse',
                'violation_type': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'violation_probability': 1.0,
                'severity_score': 0.5,
                'violation_type': 0.7
            },
            metrics={
                'violation_probability': ['accuracy', keras.metrics.AUC(name='auc')],
                'severity_score': ['mae'],
                'violation_type': ['accuracy']
            }
        )
        
        print("Model compiled!")
        print("   Multi-output configuration:")
        print("   - Violation probability (binary)")
        print("   - Severity score (regression)")
        print("   - Violation type (4 classes)")
    
    def predict_future_violation(self, image_sequence):
        """
        Predict future violation from image sequence
        
        Args:
            image_sequence: Sequence of images (sequence_length, H, W, 3)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            print("No model loaded")
            return None
        
        # Ensure correct shape
        if len(image_sequence.shape) == 4:
            image_sequence = np.expand_dims(image_sequence, 0)
        
        # Predict
        predictions = self.model.predict(image_sequence, verbose=0)
        
        # Parse outputs
        result = {
            'violation_probability': float(predictions['violation_probability'][0][0]),
            'severity_score': float(predictions['severity_score'][0][0]),
            'violation_type_probs': predictions['violation_type'][0].tolist(),
            'predicted_type': int(np.argmax(predictions['violation_type'][0]))
        }
        
        # Add interpretation
        violation_types = ['No Violation', 'Sand Mining', 'Land Encroachment', 'Vegetation Loss']
        result['predicted_type_name'] = violation_types[result['predicted_type']]
        
        # Risk level
        if result['violation_probability'] > 0.7:
            result['risk_level'] = 'HIGH'
        elif result['violation_probability'] > 0.4:
            result['risk_level'] = 'MEDIUM'
        else:
            result['risk_level'] = 'LOW'
        
        return result
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
    
    def save_model(self, filepath):
        """Save model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")


def create_synthetic_temporal_data(num_sequences=50, sequence_length=5):
    """
    Create synthetic temporal sequences for demonstration
    
    Simulates:
    - Stable areas (no violation)
    - Progressive degradation (future violation)
    - Sudden changes (immediate violation)
    
    Args:
        num_sequences: Number of temporal sequences
        sequence_length: Number of time steps per sequence
        
    Returns:
        X: Image sequences (num_sequences, sequence_length, H, W, 3)
        y: Labels dictionary
    """
    print(f"\nCreating {num_sequences} temporal sequences...")
    
    X = []
    y_violation = []
    y_severity = []
    y_type = []
    
    for seq_idx in range(num_sequences):
        sequence = []
        
        # Randomly choose scenario
        scenario = np.random.choice(['stable', 'progressive', 'sudden'], p=[0.4, 0.4, 0.2])
        
        for t in range(sequence_length):
            # Create base image
            img = np.random.randint(50, 150, (64, 64, 3), dtype=np.uint8)
            
            if scenario == 'stable':
                # No change over time
                img = img.astype(np.float32) / 255.0
                violation = 0
                severity = 0.0
                v_type = 0  # No violation
                
            elif scenario == 'progressive':
                # Gradual degradation
                degradation = (t / sequence_length) * 100
                img = (img - degradation).clip(0, 255).astype(np.uint8)
                img = img.astype(np.float32) / 255.0
                
                # Future violation predicted
                violation = 1
                severity = min(0.5 + (t / sequence_length) * 0.5, 1.0)
                v_type = np.random.choice([1, 2, 3])  # Mining, Encroachment, or Vegetation
                
            else:  # sudden
                # Sudden change in last frame
                if t == sequence_length - 1:
                    img = (img * 0.3).astype(np.uint8)
                img = img.astype(np.float32) / 255.0
                
                violation = 1
                severity = 0.8
                v_type = 1  # Mining
            
            sequence.append(img)
        
        X.append(sequence)
        y_violation.append(violation)
        y_severity.append(severity)
        y_type.append(v_type)
    
    X = np.array(X)
    y = {
        'violation_probability': np.array(y_violation, dtype=np.float32),
        'severity_score': np.array(y_severity, dtype=np.float32),
        'violation_type': np.array(y_type, dtype=np.int32)
    }
    
    print(f"Created temporal sequences")
    print(f"   Shape: {X.shape}")
    print(f"   Violation rate: {np.mean(y_violation)*100:.1f}%")
    
    return X, y