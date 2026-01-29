#test to check if image processor is working fine
from utils.image_processor import ImageProcessor

processor = ImageProcessor(target_size=256)

# Test loading an image 
print("\n Testing ImageProcessor...")
print("=" * 50)


print(f"image processor created , Target size: {processor.target_size}x{processor.target_size}")
