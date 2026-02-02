"""
Test change detection functionality
"""

from utils.image_processor import ImageProcessor
from utils.change_detector import ChangeDetector
import numpy as np
import os

# Initialize
processor = ImageProcessor(target_size=256)
change_detector = ChangeDetector(change_threshold=50)

# Load test image
test_image_path = 'uploads/test_image.jpg'

if not os.path.exists(test_image_path):
    print("❌ Test image not found")
else:
    print("=" * 70)
    print("🧪 TESTING CHANGE DETECTION - SEVAS")
    print("=" * 70)
    
    # Load and resize image
    print("\n📂 Loading 'before' image...")
    img_before = processor.load_image(test_image_path)
    
    if img_before is not None:
        img_before = processor.resize_image(img_before)
        
        # Simulate an "after" image by making some changes
        print("\n🎨 Creating simulated 'after' image...")
        print("   (Darkening bottom-right quadrant to simulate change)")
        
        img_after = img_before.copy()
        
        # Darken bottom-right quadrant to simulate land use change
        h, w = img_after.shape[:2]
        img_after[h//2:, w//2:] = (img_after[h//2:, w//2:] * 0.5).astype(np.uint8)
        
        print("   ✅ 'After' image created (simulated change)")
        
        # Detect changes
        print("\n" + "="*70)
        change_mask, change_pct, diff_img = change_detector.detect_changes(
            img_before, 
            img_after
        )
        
        if change_mask is not None:
            # Visualize changes
            print("\n" + "="*70)
            change_detector.visualize_changes(
                img_before,
                img_after,
                change_mask,
                'outputs/change_detection.png'
            )
            
            # Analyze change type
            print("\n" + "="*70)
            analysis = change_detector.analyze_change_type(
                img_before,
                img_after,
                change_mask
            )
            
            print("\n" + "=" * 70)
            print("✅ CHANGE DETECTION COMPLETE!")
            print("\n📊 Summary:")
            print(f"   Change detected: {change_pct:.2f}% of image")
            print(f"   Change type: {analysis.get('type', 'unknown')}")
            print(f"   Description: {analysis.get('description', 'N/A')}")
            print("\n📁 Generated file:")
            print("   outputs/change_detection.png")
            print("=" * 70)
            