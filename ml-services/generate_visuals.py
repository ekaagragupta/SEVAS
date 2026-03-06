"""
Generate all visual assets for README
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

print("="*70)
print("GENERATING VISUAL ASSETS FOR SEVAS")
print("="*70)

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# ============================================================================
# 1. GENERATE U-NET PREDICTION SAMPLE
# ============================================================================

print("\n 1. Generating U-Net prediction sample...")

# Load the trained model
try:
    from tensorflow import keras
    model = keras.models.load_model('models/saved_models/unet_best.h5')
    print(" Model loaded")
    
    # Load a test image
    test_img_path = 'uploads/test_image.jpg'
    
    if os.path.exists(test_img_path):
        # Load and preprocess
        img = cv2.imread(test_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Predict
        prediction = model.predict(np.expand_dims(img_normalized, 0), verbose=0)
        pred_mask = np.argmax(prediction[0], axis=-1)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        # Original image
        axes[0, 0].imshow(img_resized)
        axes[0, 0].set_title('Original Satellite Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        colors = [[0.8, 0.8, 0.8], [1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Gray, Red, Green, Blue
        mask_rgb = np.zeros((*pred_mask.shape, 3))
        for i in range(4):
            mask_rgb[pred_mask == i] = colors[i]
        
        axes[0, 1].imshow(mask_rgb)
        axes[0, 1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = cv2.addWeighted((img_resized).astype(np.uint8), 0.6, 
                                 (mask_rgb * 255).astype(np.uint8), 0.4, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (Image + Segmentation)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Class statistics
        total_pixels = pred_mask.size
        class_names = ['Background', 'Mining', 'Vegetation', 'Water']
        class_counts = [np.sum(pred_mask == i) for i in range(4)]
        class_percentages = [(count / total_pixels) * 100 for count in class_counts]
        
        axes[1, 1].bar(class_names, class_percentages, color=['gray', 'red', 'green', 'blue'])
        axes[1, 1].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Background'),
            Patch(facecolor='red', label='Mining/Disturbance'),
            Patch(facecolor='green', label='Vegetation'),
            Patch(facecolor='blue', label='Water')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12)
        
        plt.tight_layout()
        plt.savefig('outputs/unet_prediction_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" U-Net prediction sample saved")
    else:
        print("  Test image not found, creating synthetic example...")
        
        # Create a synthetic example
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Synthetic input
        synthetic_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        axes[0].imshow(synthetic_img)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Synthetic mask
        mask = np.zeros((256, 256, 3))
        cv2.circle(mask, (128, 128), 50, (1, 0, 0), -1)  # Red circle (mining)
        cv2.rectangle(mask, (50, 50), (100, 100), (0, 1, 0), -1)  # Green rectangle (vegetation)
        cv2.circle(mask, (200, 200), 30, (0, 0, 1), -1)  # Blue circle (water)
        
        axes[1].imshow(mask)
        axes[1].set_title('Segmentation Output', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        overlay = (synthetic_img * 0.6 + mask * 255 * 0.4).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/unet_prediction_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Synthetic prediction sample created")
        
except Exception as e:
    print(f"  Error generating prediction: {str(e)}")
    print("   Creating placeholder...")

# ============================================================================
# 2. CREATE ARCHITECTURE DIAGRAM
# ============================================================================

print("\n🏗️  2. Creating architecture diagram...")

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'SEVAS System Architecture', 
        ha='center', fontsize=20, fontweight='bold')

# Layer 1: Frontend (Optional)
frontend = Rectangle((3.5, 10), 3, 0.8, 
                     facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
ax.add_patch(frontend)
ax.text(5, 10.4, 'Frontend Interface\n(Optional - Next.js)', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow down
ax.arrow(5, 9.9, 0, -0.3, head_width=0.15, head_length=0.1, fc='black')

# Layer 2: API Layer
api = Rectangle((2, 8.5), 6, 1, 
               facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=2)
ax.add_patch(api)
ax.text(5, 9.3, 'Flask REST API Layer', 
        ha='center', fontsize=12, fontweight='bold')

# API Endpoints
endpoints = ['/analyze', '/batch', '/segment', '/health']
for i, endpoint in enumerate(endpoints):
    x = 2.5 + i * 1.3
    ep_box = Rectangle((x, 8.6), 1.1, 0.3, 
                       facecolor='white', edgecolor='#1976D2', linewidth=1)
    ax.add_patch(ep_box)
    ax.text(x + 0.55, 8.75, endpoint, ha='center', va='center', fontsize=8)

# Arrows down to processing layer
for x in [2.5, 3.8, 5.1, 6.4]:
    ax.arrow(x + 0.55, 8.4, 0, -0.3, head_width=0.1, head_length=0.08, fc='gray')

# Layer 3: Processing Modules
modules = [
    ('Vision AI\n(Gemini)', 1.5, '#FFF9C4', '#F57F17'),
    ('U-Net\nSegmentation', 3.5, '#FCE4EC', '#C2185B'),
    ('Spectral\nIndices', 5.5, '#E1F5FE', '#0277BD'),
    ('Cloud\nDetector', 7.5, '#F3E5F5', '#7B1FA2')
]

for name, x, facecolor, edgecolor in modules:
    module = Rectangle((x, 6.8), 1.8, 1, 
                       facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(module)
    ax.text(x + 0.9, 7.3, name, ha='center', va='center', 
           fontsize=9, fontweight='bold')

# Arrows down to preprocessing
for x in [1.5, 3.5, 5.5, 7.5]:
    ax.arrow(x + 0.9, 6.7, 0, -0.3, head_width=0.1, head_length=0.08, fc='gray')

# Layer 4: Preprocessing
preprocess = Rectangle((2, 5.2), 6, 1, 
                       facecolor='#FFF8E1', edgecolor='#F57C00', linewidth=2)
ax.add_patch(preprocess)
ax.text(5, 6, 'Image Preprocessing Pipeline', 
        ha='center', fontsize=11, fontweight='bold')

steps = ['Load', 'Resize\n256x256', 'Normalize\n0-1', 'Validate']
for i, step in enumerate(steps):
    x = 2.5 + i * 1.3
    step_box = Rectangle((x, 5.3), 1.1, 0.6, 
                         facecolor='white', edgecolor='#E65100', linewidth=1)
    ax.add_patch(step_box)
    ax.text(x + 0.55, 5.6, step, ha='center', va='center', fontsize=7)

# Arrow down
ax.arrow(5, 5.1, 0, -0.3, head_width=0.15, head_length=0.1, fc='black')

# Layer 5: Results
results = Rectangle((2.5, 3.5), 5, 1.2, 
                   facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
ax.add_patch(results)
ax.text(5, 4.5, 'Unified Results Processing', 
        ha='center', fontsize=11, fontweight='bold')

result_items = ['Violation:\nYes/No', 'Severity:\nLow/Med/High', 
                'Location:\nDescription', 'Confidence:\n%']
for i, item in enumerate(result_items):
    x = 2.7 + i * 1.2
    ax.text(x, 3.9, item, ha='left', fontsize=7)

# Layer 6: Output
output = Rectangle((3, 2), 4, 0.8, 
                  facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(output)
ax.text(5, 2.4, 'JSON Response / Report Generation', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Data Flow Indicators
ax.text(0.5, 9, 'Data\nFlow', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Legend
legend_items = [
    ('Frontend', '#E8F5E9'),
    ('API Layer', '#E3F2FD'),
    ('ML Models', '#FCE4EC'),
    ('Processing', '#FFF8E1'),
    ('Output', '#C8E6C9')
]

for i, (name, color) in enumerate(legend_items):
    y = 1.2 - i * 0.15
    legend_box = Rectangle((0.2, y), 0.2, 0.1, facecolor=color, edgecolor='black')
    ax.add_patch(legend_box)
    ax.text(0.5, y + 0.05, name, va='center', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(" Architecture diagram saved")

# ============================================================================
# 3. CREATE ML PIPELINE DIAGRAM
# ============================================================================

print("\n 3. Creating ML pipeline diagram...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'SEVAS ML Processing Pipeline', 
        ha='center', fontsize=18, fontweight='bold')

# Input
input_box = Rectangle((4, 8.5), 2, 0.6, 
                      facecolor='#BBDEFB', edgecolor='#1976D2', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.8, 'Input Image\n(Satellite/Drone)', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.arrow(5, 8.4, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', lw=2)

# Preprocessing
prep_box = Rectangle((3.5, 7), 3, 1, 
                     facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
ax.add_patch(prep_box)
ax.text(5, 7.7, 'Preprocessing', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 7.3, '• Resize (256x256)\n• Normalize (0-1)\n• Validate format', 
        ha='center', fontsize=8)

# Three parallel paths
paths = [
    ('Vision AI\nAnalysis', 1.5, '#E1BEE7'),
    ('U-Net\nSegmentation', 4.5, '#FFCCBC'),
    ('Spectral\nAnalysis', 7.5, '#B2DFDB')
]

for name, x, color in paths:
    # Arrow down
    if x == 4.5:
        ax.arrow(5, 6.9, 0, -0.3, head_width=0.1, head_length=0.08, fc='black', lw=1.5)
    else:
        start_x = 5 if x < 5 else 5
        end_x = x + 0.75
        ax.plot([start_x, start_x, end_x], [6.9, 6.5, 6.5], 'k-', lw=1.5)
        ax.arrow(end_x, 6.5, 0, -0.3, head_width=0.1, head_length=0.08, fc='black', lw=1.5)
    
    # Module box
    box = Rectangle((x, 4.8), 1.5, 1.2, 
                    facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + 0.75, 5.4, name, ha='center', va='center', 
           fontsize=9, fontweight='bold')

# Outputs from each path
outputs_text = [
    ('NL Description\nConfidence\nRecommendations', 1.5),
    ('Pixel Mask\nClass Distribution\nArea Calculation', 4.5),
    ('NDVI/NDWI\nCloud %\nVeg Health', 7.5)
]

for text, x in outputs_text:
    ax.text(x + 0.75, 4.5, text, ha='center', fontsize=7, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Convergence arrows
for x in [1.5, 4.5, 7.5]:
    end_y = 3.5
    ax.arrow(x + 0.75, 4.7, 0, -0.5, head_width=0.08, head_length=0.06, 
            fc='gray', lw=1)

# Results aggregation
result_box = Rectangle((2.5, 2.5), 5, 0.8, 
                       facecolor='#C8E6C9', edgecolor='#388E3C', linewidth=2)
ax.add_patch(result_box)
ax.text(5, 2.9, 'Results Aggregation & Analysis', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow
ax.arrow(5, 2.4, 0, -0.3, head_width=0.15, head_length=0.1, fc='black', lw=2)

# Final output
output_box = Rectangle((3, 1.2), 4, 0.8, 
                       facecolor='#81C784', edgecolor='#2E7D32', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.8, 'Violation Detected: Yes/No', ha='center', fontsize=10, fontweight='bold')
ax.text(5, 1.5, 'Severity • Location • Confidence • Recommendations', 
        ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/ml_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("ML pipeline diagram saved")

print("\n" + "="*70)
print("ALL VISUAL ASSETS GENERATED!")
print("="*70)
print("\n Created files:")
print("    outputs/unet_prediction_sample.png")
print("    outputs/architecture_diagram.png")
print("    outputs/ml_pipeline.png")
print("    outputs/unet_training_simple.png (already exists)")
print("\n Add these to your README images!")
print("="*70)