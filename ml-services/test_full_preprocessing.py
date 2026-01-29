#test for complete pipeline preprocessing 
from utils.image_processor import ImageProcessor
import os 

processor = ImageProcessor(target_size=256)

test_image_path='ml-services/uploads/test_image.jpg'

if not os.path.exists(test_image_path):
    print(f"Test image not found at {test_image_path}. Please ensure the test image is available.")
else:
    preprocessed=processor.preprocess_image(test_image_path)
    
    if preprocessed is not None:
        #savinbg  comparison 
        processor.save_preprocessed_comparison(
            test_image_path,
            preprocessed,
            'ml-services/outputs/preprocessed_comparison.jpg'
        )
        print("preprocessed comparison is now saved")