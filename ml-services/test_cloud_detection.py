from utils.image_processor import ImageProcessor
from utils.cloud_detector import CloudDetector
import os
import cv2
import numpy as np

processor = ImageProcessor(target_size=256)
cloud_detector = CloudDetector(brightness_threshold=200, cloud_threshold_percent=30)

# Load test image
test_image_path = 'uploads/test_image.jpg'

if not os.path.exists(test_image_path):
    print(" Test image not found")
else:
    print("=" * 70)
    print(" TESTING CLOUD DETECTION - SEVAS")
    print("=" * 70)
    
   
    print("\n Loading image...")
    img = processor.load_image(test_image_path)
    
    if img is not None:
        print("\n Resizing image...")
        img_resized = processor.resize_image(img)
        
        if img_resized is not None:
        
            print("\n" + "="*70)
            cloud_mask, cloud_percentage = cloud_detector.detect_clouds(img_resized)
            
            if cloud_mask is not None:
            
                is_usable = cloud_detector.is_image_usable(cloud_percentage)
                
                # Visualize clouds
                
                cloud_detector.visualize_cloud_mask(
                    img_resized,
                    cloud_mask,
                    'outputs/cloud_detection.png'
                )
                
                print("\n" + "=" * 70)
                print(" CLOUD DETECTION COMPLETE!")
                print("\n Results:")
                print(f"   Cloud coverage: {cloud_percentage:.2f}%")
                print(f"   Image usable: {'Yes ' if is_usable else 'No '}")
                print("\n Generated file:")
                print("   outputs/cloud_detection.png (clouds highlighted in red)")
                