#feeding all image processing before sending to ML model

import cv2
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