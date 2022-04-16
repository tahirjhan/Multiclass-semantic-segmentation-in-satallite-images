"""
Dataset downloaded from
https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery

The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:

Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

Use patchify....
Tile 1: 797 x 644 --> 768 x 512 --> 6
Tile 2: 509 x 544 --> 512 x 256 --> 2
Tile 3: 682 x 658 --> 512 x 512  --> 4
Tile 4: 1099 x 846 --> 1024 x 768 --> 12
Tile 5: 1126 x 1058 --> 1024 x 1024 --> 16
Tile 6: 859 x 838 --> 768 x 768 --> 9
Tile 7: 1817 x 2061 --> 1792 x 2048 --> 56
Tile 8: 2149 x 1479 --> 1280 x 2048 --> 40
Total 9 images in each folder * (145 patches) = 1305
Total 1305 patches of size 256x256

"""
import os
import cv2
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
#import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

#process dataset
root_directory = 'raw_data/'
save_directory_imgs = 'processed_dataset/images'
save_directory_masks = 'processed_dataset/masks'
image_counter=0
masks_counter=0
####################
# Patch size of train images to train network
patch_size = 256

#Read images from repsective 'images' subdirectory
#As all images are of different size we have 2 options, either resize or crop
#But, some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then
#divide all images into patches of 256x256x3.

image_dataset = []

# Loop through images in sub folder, and extract patches one by one.
for path, subdirs, files in os.walk(root_directory):
   # print(files) # cross check if files read successfully
    dirname = path.split(os.path.sep)[-1]  # select the path and read only images
    if dirname == 'images':  # Find all 'images' directories
        images = os.listdir(path)  # List of all image names in this subdirectory
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"):  # Only read jpg images.
                image = cv2.imread(path+"/"+image_name,1) # read imaga as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                image = Image.fromarray(image) # Creates an image memory from an object exporting the array interface (using the buffer protocol).
                image = image.crop((0,0,SIZE_X,SIZE_Y)) #Crop from top left corner
                image = np.array(image)
                window_name = 'image'
                # cv2.imshow(window_name, image)
                # #cv2.imshow(image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Extract patches from single image read above using patchify
                print('Patchifying image:', path+'/'+image_name)

                # Apply patchify
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)

