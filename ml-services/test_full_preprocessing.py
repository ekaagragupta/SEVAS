"""
Test complete preprocessing pipeline
"""

import os
import sys

# Add the parent directory to Python path
# This allows Python to find the 'utils' folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_processor import ImageProcessor

# Create processor
processor = ImageProcessor(target_size=256)

# Path to test image
test_image_path = 'ml-services/uploads/test_image.jpg'

# Check if file exists
if not os.path.exists(test_image_path):
    print(f"❌ Test image not found at {test_image_path}. Please ensure the test image is available.")
    print("   You can use any river/landscape satellite image from Google Images")
else:
    # Preprocess the image
    preprocessed = processor.preprocess_image(test_image_path)
    
    if preprocessed is not None:
        # Save comparison
        processor.save_preprocessed_comparison(
            test_image_path,
            preprocessed,
            'ml-services/outputs/preprocessed_comparison.jpg'
        )
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Check ml-services/outputs/preprocessed_comparison.jpg")
        print("="*60)
