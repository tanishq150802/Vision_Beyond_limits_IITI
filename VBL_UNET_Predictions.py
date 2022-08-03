import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

IMG_SIZE = 512
train_images = []
image_names = []

for img_name in os.listdir("VBL_Images"):
        image_names.append(img_name)
        img = cv2.imread(f"VBL_Images/{img_name}")       
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        train_images.append(img)

train_images = np.array(train_images)

train_images = train_images.reshape(-1,IMG_SIZE,IMG_SIZE,3)
train_images = train_images/255.0

X = train_images

model = tf.keras.models.load_model("VBL_UNET_Model.hdf5")

color_coding = {'background': (255, 255, 255), #white 
                'no-damage': (0,255,0), #green
                'minor-damage': (255,255,0), #yellow
                'major-damage': (255,165,0), #orange
                'destroyed': (255,0,0), #red
                'un-classified': (0,0,0) #black
                } 

predictions = model.predict(X)

for i in range(117):
        single_prediction = predictions[i]
        predicted_masks = np.argmax(single_prediction,axis=2)

        segmented_image = np.ones((IMG_SIZE, IMG_SIZE, 3))  

        segmented_image[predicted_masks == 0] = color_coding['background']
        segmented_image[predicted_masks == 1] = color_coding['no-damage']
        segmented_image[predicted_masks == 2] = color_coding['minor-damage']
        segmented_image[predicted_masks == 3] = color_coding['major-damage']
        segmented_image[predicted_masks == 4] = color_coding['destroyed']
        segmented_image[predicted_masks == 5] = color_coding['un-classified']

        
        im = Image.fromarray(segmented_image.astype(np.uint8))
        im.save(f"VBL_Segmented_Images/{image_names[i]}")