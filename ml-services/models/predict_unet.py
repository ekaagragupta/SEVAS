
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from unet_model import UNetModel

class UNetPredictor:
    """
    Make predictions using trained U-Net model
    """
    
    def __init__(self, model_path='models/saved_models/unet_best.h5'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
        """
        print(" Loading U-Net model...")
        
        self.unet = UNetModel()
        self.unet.load_model(model_path)
        
        # Define class colors for visualization
        self.class_colors = {
            0: [200, 200, 200],  # Background - Gray
            1: [255, 0, 0],      # Mining - Red
            2: [0, 255, 0],      # Vegetation - Green
            3: [0, 0, 255]       # Water - Blue
        }
        
        self.class_names = {
            0: 'Background',
            1: 'Sand Mining',
            2: 'Vegetation',
            3: 'Water'
        }
        
        print(" Model loaded successfully!")
    
    def predict_image(self, image_path):
        """
        Predict segmentation for an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            mask: Segmentation mask
            confidence: Confidence map
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image_normalized = image.astype(np.float32) / 255.0
        
        # Predict
        mask, confidence = self.unet.predict_with_confidence(image_normalized)
        
        return image, mask, confidence
    
    def mask_to_rgb(self, mask):
        """
        Convert class mask to RGB image
        
        Args:
            mask: Segmentation mask (H, W) with class labels
            
        Returns:
            RGB image (H, W, 3)
        """
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            rgb[mask == class_id] = color
        
        return rgb
    
    def visualize_prediction(self, image, mask, confidence, save_path=None):
        """
        Create comprehensive visualization
        
        Args:
            image: Original image
            mask: Predicted segmentation mask
            confidence: Confidence map
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation mask
        mask_rgb = self.mask_to_rgb(mask)
        axes[0, 1].imshow(mask_rgb)
        axes[0, 1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, mask_rgb, 0.4, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Confidence map
        im = axes[1, 1].imshow(confidence, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Confidence Map', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(self.class_colors[i])/255, label=self.class_names[i])
            for i in range(len(self.class_names))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Visualization saved to: {save_path}")
        
        plt.show()
    
    def analyze_prediction(self, mask):
        """
        Analyze the prediction and generate statistics
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with analysis results
        """
        total_pixels = mask.size
        
        analysis = {
            'total_pixels': total_pixels,
            'class_distribution': {}
        }
        
        print("\n📊 Segmentation Analysis:")
        print("="*70)
        
        for class_id, class_name in self.class_names.items():
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            
            analysis['class_distribution'][class_name] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }
            
            print(f"{class_name:15s}: {count:7d} pixels ({percentage:5.2f}%)")
        
        print("="*70)
        
        # Detect violations
        mining_pct = analysis['class_distribution']['Sand Mining']['percentage']
        
        if mining_pct > 5:
            print(f"  WARNING: Sand mining detected ({mining_pct:.2f}% of image)")
            analysis['violation_detected'] = True
            analysis['severity'] = 'High' if mining_pct > 15 else 'Medium'
        else:
            print(" No significant violations detected")
            analysis['violation_detected'] = False
            analysis['severity'] = 'None'
        
        return analysis


def demo_prediction():
    """
    Demo prediction on test image
    """
    print("="*70)
    print("🔮 U-NET PREDICTION DEMO")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = UNetPredictor('models/saved_models/unet_best.h5')
    except:
        print("⚠️  Model not found. Train the model first using train_unet.py")
        return
    
    # Predict on test image
    test_image_path = 'uploads/test_image.jpg'
    
    print(f"\n Analyzing image: {test_image_path}")
    
    try:
        image, mask, confidence = predictor.predict_image(test_image_path)
        
        # Visualize
        predictor.visualize_prediction(
            image, mask, confidence,
            save_path='outputs/unet_prediction.png'
        )
        
        # Analyze
        analysis = predictor.analyze_prediction(mask)
        
        print("\n Prediction complete!")
        
    except Exception as e:
        print(f" Error: {str(e)}")


if __name__ == '__main__':
    demo_prediction()