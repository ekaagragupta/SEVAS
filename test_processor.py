"""
Test file to check if our ImageProcessor works
"""

from utils.image_processor import ImageProcessor

processor = ImageProcessor(target_size=256)

# Test loading an image 
print("\n Testing ImageProcessor...")
print("=" * 50)

print("ImageProcessor class created successfully!")
print(f"   Target size: {processor.target_size}x{processor.target_size}")
