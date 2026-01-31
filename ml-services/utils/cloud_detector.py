import numpy as np
import cv2

class CloudDetector:
    def __init__(self,brightness_threshold=200,cloud_threshold_percent=30):
        self.brightness_threshold = brightness_threshold
        self.cloud_threshold_percent = cloud_threshold_percent
        
        print(f"check point cloud detector initisalized")
        print(f"Brightness Threshold: {self.brightness_threshold}")
        print(f"Cloud Threshold Percent: {self.cloud_threshold_percent}")   
        
    def detect_clouds(self,image):
        print("check point detecting clouds...")
        try:
            img=image.copy()
            #checking image is in range 0-255
            if img.max()<=1.0:
                img = (img * 255).astype(np.uint8)
            if len(img.shape)==3:
                red=img[:,:,2]
                green=img[:,:,1]
                blue=img[:,:,0]
            else:
                print("check point image is not in expected format")
                return None
            '''
    detection logic : clouds are bright for all channels , im creating 
    boolean marks for each channel based on brightness threshold
    '''
            red_bright = red > self.brightness_threshold
            green_bright = green > self.brightness_threshold
            blue_bright = blue > self.brightness_threshold
            
            '''
            # Example with threshold = 200:

red channel = [[150, 230, 240],
               [180, 250, 200],
               [245, 255, 190]]

red_bright = [[False, True,  True ],
              [False, True,  False],
              [True,  True,  False]]
              
              false is eliminated as not bright
                '''
            
            # ek pixel = all channels bright
            cloud_mask = red_bright & green_bright & blue_bright
            cloud_mask=cloud_mask.astype(np.uint8)
            
            total_pixels=cloud_mask.size
            cloud_pixels=np.sum(cloud_mask)
            cloud_percentage=float((cloud_pixels/total_pixels)*100)
            
            print(f"cloud percentage: {cloud_percentage}%")
            print(f"total pixels: {total_pixels}, cloud pixels: {cloud_pixels} ")
            
            #result 
            if cloud_percentage < 10:
                print(" Status : Clear sky (<10% clouds)")
            elif cloud_percentage < self.cloud_threshold_percent:
                print(f"Status : Acceptable clouds ({cloud_percentage:.2f}% clouds)")
            else:
                print(f"Status : Too cloudy ({cloud_percentage:.2f}% clouds)")
            return cloud_percentage, cloud_mask
        except Exception as e:
            print(f"Error in cloud detection: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def visualize_clouds(self,image,cloud_mask,output_path):
        try :
            print("check point visualizing clouds....")
            if image.max()<=1.0:
                image = (image * 255).astype(np.uint8)
            vis_image=image.copy()
            
            # cloud mask =1 then highlight in red cloud areas
            red_overlay= np.zeros_like(vis_image)
            red_overlay[:,:,2]=255
            # blending 70 % og image and 30% red overlay
            
            alpha=0.3
            
            cloud_mask_3d=np.stack([cloud_mask,cloud_mask,cloud_mask],axis=2)
            vis_image=np.where(cloud_mask_3d==1
                               , cv2.addWeighted(vis_image,1-alpha,red_overlay,alpha,0),
                               vis_image)
            
            #save 
            cv2.imwrite(output_path,vis_image)
            print(f"Cloud visualization saved to {output_path}")
        except Exception as e:
            print(f"Error in visualizing clouds: {e}")  
            
    def is_image_usable(self,cloud_percentage):
        if isinstance(cloud_percentage, np.ndarray):
              cloud_percentage = float(cloud_percentage)
        usable = float(cloud_percentage) < self.cloud_threshold_percent
        if usable:
            print("Image is usable for further processing.")
        else:
            print("Image is not usable due to high cloud coverage.")
        return usable