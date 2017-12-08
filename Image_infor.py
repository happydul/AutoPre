from scipy.misc import imread
import os
file_dir = os.path.join('data','driving_data','train')
image_dir = os.path.join(file_dir,'0.jpg')
source_image = imread(image_dir)
print source_image.shape



