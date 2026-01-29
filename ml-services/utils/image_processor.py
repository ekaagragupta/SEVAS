#feeding all image processing before sending to ML model
import numpy as np
#Opens JPG/PNG files
from PIL import Image
import os

#class to handle image processing 
#basically photo editor before sending to AI model
class ImageProcessor:
    def __init__(self,target_size=256):
        self.target_size = target_size
        print("ImageProcessor initialized with target size:", self.target_size)

    def load_image(self,image_path):
            try:
                if not os.path.exists(image_path):
                    print("File does not exist:", image_path)
                    return None
                img = Image.open(image_path)
                #converting to same format 
                img=img.convert('RGB')
                #image to numbers  conversion using numpy
                img_array=np.array(img)
                print("Image loaded : {img_array.shape}")
                return img_array
            except Exception as e:
                print("Error loading image:", e)
                return None
    
    def resize_image(self,img_array):
        
            try:
                #current dimensions
                current_height, current_width = img_array.shape[:2]

                if current_width>self.target_size or current_height>self.target_size:
                     #shrink image
                     interpolation = cv2.INTER_AREA
                else:
                     #enlarge image
                     interpolation = cv2.INTER_CUBIC
                
                resized= cv2.resize(
                     img_array,
                        (self.target_size, self.target_size), 
                        # interpolation method based on shrinking or enlarging  
                        interpolation=interpolation
                )
                print("image resized to:{self.target_size}*{self.target_size}")
                return resized
            except Exception as e:
                print("Error resizing image:", e)
                return None
    
    def normalize_image(self,img_array):
            #ml training is better with small no (0-1)
            try:
                img_float=img_array.astype(np.float32)
                normalized=img_float/255.0
                print("Image normalized.")
                return normalized
            except Exception as e:
                print("Error normalizing image:", e)
                return None
    def preprocess_image(self,image_path):
            #full pipeline 
            '''
            1. load 2. resize 3. normalize
            '''
            print('starting preprocessing for image:', image_path)
            img=self.load_image(image_path)

            if img is None:
                return None
            
            img_resized=self.resize_image(img)
            if img_resized is None:
                return None
            
            print('preprocessing completed')
            print(f"final shape:{img_resized.shape}")
            img_normalized=self.normalize_image(img_resized)
            print(f"value ranges from :[{img_normalized.min():.3f}]")
            return img_normalized
    
    #for processing multiple images at once
    def preprocess_batch(self,image_paths):
            print(f'batch preprocessing for {len(image_paths)}')
            preprocessed_images=[]

            for i,path in enumerate(image_paths,1):
                 print(f"\n...Image{i}/{len(image_paths)}...")
                 img=self.preprocess_image(path)

                 if img is not None :
                     preprocessed_images.append(img)
                 else:
                     print("error")
            if len(preprocessed_images)==0:
                print("no image found")
                return None
            batch =np.array(preprocessed_images)
            print(f"batch prep completed")
            print(f"batch shape : {batch.shape}")
            print(f"successful preprocessed images : {len(preprocessed_images)}/{len(image_paths)}")
            return batch

    def save_preprocessed_comparison(self, original_path, preprocessed_array, output_path):
        """
        Save before/after comparison image
        """
        try:
            import matplotlib.pyplot as plt
            
            # Load original
            original = self.load_image(original_path)
            
            # Convert preprocessed back to 0-255 for display
            # (multiply by 255 and convert to uint8)
            preprocessed_display = (preprocessed_array * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Show original
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Show preprocessed
            axes[1].imshow(preprocessed_display)
            axes[1].set_title(f'Preprocessed ({self.target_size}x{self.target_size}, Normalized)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f" Comparison saved to: {output_path}")
            
        except Exception as e:
            print(f"  Could not save comparison: {str(e)}")