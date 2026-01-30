import numpy as np
import cv2

class SpectralIndices:
    # RGB images --> simulate bands
    #real stat data (GeoTIFF)--> use actual bands
    def __init__(self):
        print("Spectral Indices Utility Initialized")
        
    def extract_rgb_bands(self,image):
        #image shape =(height,width,channels) = (256,256,3)
        if len(image.shape)==3 and image.shape[2]==3:
            blue=image[:,:,0]
            green=image[:,:,1]
            red=image[:,:,2]
            return blue, green, red
        else:
            raise ValueError("Input image must be an RGB image with 3 channels.")
        
    def calculate_ndvi(self,image,nir_band=None):
        print("Calculating NDVI..")
        try:
            red,green,blue=self.extract_rgb_bands(image)
            if red is None:
                return None
            if nir_band is not None:
                nir=nir_band
                print("using real NIR band")
            else:
                nir=green
                print("simulating NIR band using Green channel")
            nir=nir.astype(np.float32)
            red=red.astype(np.float32)
            
            epsilon=1e-10
            ndvi=(nir-red)/(nir+red+epsilon) #NDVI between -1 to 1
            ndvi = np.clip(ndvi, -1, 1)
            
            mean_ndvi=ndvi.mean()
            if mean_ndvi>0.6:
                print("🌲 Interpretation: Dense vegetation detected")
            elif mean_ndvi > 0.2:
                print(f"   🟡 Interpretation: Moderate vegetation")
            elif mean_ndvi > -0.1:
                print(f"   🟠 Interpretation: Bare soil/sparse vegetation")
            else:
                print(f"   🔵 Interpretation: Water/clouds/snow")
            
            return ndvi
            
        except Exception as e:
            print(f"❌ Error calculating NDVI: {str(e)}")
            return None  
    
    def calculate_ndwi(self, image):
        
        print("\n Calculating NDWI...")
        
        try:
            # Extract bands
            red, green, blue = self.extract_rgb_bands(image)
            
            if red is None:
                return None
            
            # Approximate NIR with Red for RGB images
            nir = red
            print("  Approximating NIR with Red band (RGB image)")
            
            # Convert to float
            green = green.astype(np.float32)
            nir = nir.astype(np.float32)
            
            # Calculate NDWI
            epsilon = 1e-10
            ndwi = (green - nir) / (green + nir + epsilon)
            
            # Clip to valid range
            ndwi = np.clip(ndwi, -1, 1)
            
            # Print statistics
            print(f"   NDWI Stats:")
            print(f"   - Min: {ndwi.min():.3f}")
            print(f"   - Max: {ndwi.max():.3f}")
            print(f"   - Mean: {ndwi.mean():.3f}")
            
            # Interpret mean NDWI
            mean_ndwi = ndwi.mean()
            if mean_ndwi > 0.3:
                print(f"   💧 Interpretation: Water bodies present")
            elif mean_ndwi > -0.3:
                print(f"   🌍 Interpretation: Vegetation/land")
            else:
                print(f"   🏜️  Interpretation: Dry/bare land")
            
            return ndwi
            
        except Exception as e:
            print(f" Error calculating NDWI: {str(e)}")
            return None
    
    def visualise_index(self,index_array,output_path,index_name="Index",colormap=cv2.COLORMAP_JET):
        try:
            ## NDVI/NDWI range: -1 to +1
                          # Image display range: 0 to 255
            normalized=((index_array+1)/2*255).astype(np.uint8)
            colored=cv2.applyColorMap(normalized,colormap)
            cv2.imwrite(output_path,colored)
            print(f"{index_name} visualization saved to {output_path}")
        except Exception as e:
            print(f"Error visualizing {index_name}: {str(e)}")
            
            '''
Colormap JET:
0-50:    Blue (low values, water/bare land)
50-100:  Cyan
100-150: Green (medium vegetation)
150-200: Yellow
200-255: Red (high values, dense vegetation)
            '''