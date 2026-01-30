"""
Test spectral indices calculation
"""

from utils.image_processor import ImageProcessor
from utils.spectral_indices import SpectralIndices
import cv2
import os

processor = ImageProcessor(target_size=256)
spectral = SpectralIndices()

test_image_path = 'uploads/test_image.jpg'

if not os.path.exists(test_image_path):
    print(" Please add test_image.jpg to uploads/ folder")
else:
    print("=" * 70)
    print("TESTING SPECTRAL INDICES")
    print("=" * 70)
    
    # not normalized, we need 0-255 range
    img = processor.load_image(test_image_path)
    img_resized = processor.resize_image(img)
    
    if img_resized is not None:
        
        ndvi = spectral.calculate_ndvi(img_resized)
        
        if ndvi is not None:
            
            spectral.visualize_index(
                ndvi,
                'outputs/ndvi_visualization.png',
                index_name="NDVI",
                colormap=cv2.COLORMAP_JET
            )
        
      
        ndwi = spectral.calculate_ndwi(img_resized)
        
        if ndwi is not None:
            # Visualize NDWI
            spectral.visualize_index(
                ndwi,
                'outputs/ndwi_visualization.png',
                index_name="NDWI",
                colormap=cv2.COLORMAP_OCEAN
            )
        
        print("\n" + "=" * 70)
        print("✅ TESTING COMPLETE!")
        print("Check outputs/ folder for:")
        print("  - ndvi_visualization.png (vegetation health)")
        print("  - ndwi_visualization.png (water detection)")