
import numpy as np
import cv2

class ChangeDetector:
    """
    Detects changes between two temporal images
    Useful for tracking sand mining, deforestation, construction
    """
    
    def __init__(self, change_threshold=50):
        """
        Initialize change detector
        
        Args:
            change_threshold (int): Pixel difference above this = changed (0-255)
        """
        self.change_threshold = change_threshold
        
        print(f"ChangeDetector initialized")
        print(f"   Change threshold: {change_threshold}")
    
    def detect_changes(self, image_before, image_after):
        """
        Detect pixel-level changes between two images
        
        Args:
            image_before (numpy.ndarray): Earlier image (0-255 or 0-1)
            image_after (numpy.ndarray): Later image (0-255 or 0-1)
            
        Returns:
            tuple: (change_mask, change_percentage, difference_image)
                - change_mask: Binary mask (1=changed, 0=unchanged)
                - change_percentage: Percentage of image that changed
                - difference_image: Absolute difference between images
        """
        print("\n Detecting changes between images...")
        
        try:
            # Make copies
            img1 = image_before.copy()
            img2 = image_after.copy()
            
            # Ensure both images are in 0-255 range
            if img1.max() <= 1.0:
                img1 = (img1 * 255).astype(np.uint8)
            if img2.max() <= 1.0:
                img2 = (img2 * 255).astype(np.uint8)
            
            # Check if images have same shape
            if img1.shape != img2.shape:
                print(f" Images have different shapes!")
                print(f"   Before: {img1.shape}, After: {img2.shape}")
                print(f"   Resizing 'after' image to match 'before'...")
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Calculate absolute difference
            # For RGB images, calculate difference for each channel
            if len(img1.shape) == 3:
                # Calculate per-channel difference
                diff_r = np.abs(img1[:, :, 2].astype(np.float32) - img2[:, :, 2].astype(np.float32))
                diff_g = np.abs(img1[:, :, 1].astype(np.float32) - img2[:, :, 1].astype(np.float32))
                diff_b = np.abs(img1[:, :, 0].astype(np.float32) - img2[:, :, 0].astype(np.float32))
                
                # Average difference across channels
                difference = (diff_r + diff_g + diff_b) / 3.0
            else:
                # Grayscale image
                difference = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
            
            # Create change mask (binary: changed or not)
            change_mask = (difference > self.change_threshold).astype(np.uint8)
            
            # Calculate change percentage
            total_pixels = int(change_mask.size)
            changed_pixels = int(np.sum(change_mask))
            change_percentage = (changed_pixels / total_pixels) * 100.0
            
            print(f"   Change statistics:")
            print(f"   - Total pixels: {total_pixels:,}")
            print(f"   - Changed pixels: {changed_pixels:,}")
            print(f"   - Change percentage: {change_percentage:.2f}%")
            print(f"   - Max difference: {difference.max():.1f}")
            print(f"   - Mean difference: {difference.mean():.1f}")
            
            # Interpret results
            if change_percentage < 5:
                print(f"   ✅ Minimal change detected (< 5%)")
            elif change_percentage < 20:
                print(f"   🟡 Moderate change detected ({change_percentage:.1f}%)")
            else:
                print(f"   ⚠️  Significant change detected ({change_percentage:.1f}%)")
            
            return change_mask, change_percentage, difference.astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Error detecting changes: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, 0.0, None
    
    def visualize_changes(self, image_before, image_after, change_mask, output_path):
        """
        Create side-by-side visualization with changes highlighted
        
        Args:
            image_before: Earlier image
            image_after: Later image
            change_mask: Binary mask of changed areas
            output_path: Where to save visualization
        """
        try:
            print("\n🎨 Creating change visualization...")
            
            # Prepare images
            img1 = image_before.copy()
            img2 = image_after.copy()
            
            if img1.max() <= 1.0:
                img1 = (img1 * 255).astype(np.uint8)
            if img2.max() <= 1.0:
                img2 = (img2 * 255).astype(np.uint8)
            
            # Ensure change_mask is 2D
            if len(change_mask.shape) > 2:
                change_mask = change_mask[:, :, 0]
            
            # Create highlighted version of "after" image
            img2_highlighted = img2.copy()
            
            # Overlay changes in red
            red_overlay = np.zeros_like(img2_highlighted)
            red_overlay[:, :, 2] = 255  # Red channel
            
            alpha = 0.4
            change_mask_3d = np.stack([change_mask, change_mask, change_mask], axis=2)
            img2_highlighted = np.where(
                change_mask_3d == 1,
                cv2.addWeighted(img2_highlighted, 1-alpha, red_overlay, alpha, 0),
                img2_highlighted
            )
            
            # Create side-by-side comparison
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Before image
            axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Before', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # After image
            axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            axes[1].set_title('After', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Changes highlighted
            axes[2].imshow(cv2.cvtColor(img2_highlighted, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Changes Highlighted (Red)', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Change visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def analyze_change_type(self, image_before, image_after, change_mask):
        """
        Analyze what type of change occurred (vegetation loss, construction, etc.)
        
        Args:
            image_before: Earlier image
            image_after: Later image
            change_mask: Binary mask of changed areas
            
        Returns:
            dict: Analysis results
        """
        print("\n📊 Analyzing change type...")
        
        try:
            # Get changed pixels from both images
            changed_before = image_before[change_mask == 1]
            changed_after = image_after[change_mask == 1]
            
            if len(changed_before) == 0:
                print("   No significant changes detected")
                return {"type": "none", "confidence": 0}
            
            # Calculate average RGB values in changed areas
            avg_before = np.mean(changed_before, axis=0)
            avg_after = np.mean(changed_after, axis=0)
            
            # Simple heuristics for change type
            # Green loss = vegetation removal
            green_change = avg_after[1] - avg_before[1]  # Green channel
            brightness_change = np.mean(avg_after) - np.mean(avg_before)
            
            analysis = {}
            
            if green_change < -30:
                analysis["type"] = "vegetation_loss"
                analysis["description"] = "Possible deforestation or land clearing"
            elif brightness_change > 30:
                analysis["type"] = "construction"
                analysis["description"] = "Possible new construction or excavation"
            elif brightness_change < -30:
                analysis["type"] = "water_increase"
                analysis["description"] = "Possible flooding or water accumulation"
            else:
                analysis["type"] = "general_change"
                analysis["description"] = "Land use change detected"
            
            print(f"   Change type: {analysis['type']}")
            print(f"   Description: {analysis['description']}")
            print(f"   Green channel change: {green_change:.1f}")
            print(f"   Brightness change: {brightness_change:.1f}")
            
            return analysis
            
        except Exception as e:
            print(f" Error analyzing change type: {str(e)}")
            return {"type": "unknown", "confidence": 0}