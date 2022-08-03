import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image as im


def Create_Data(small_size):
    small_mask_values = dict()

    with open('Train_Masks_1024.json', 'r') as openfile:
        json_object = json.load(openfile)

    json_object = json_object.replace("\'", "\"")
    masks = json.loads(json_object)

    i=0

    for img_name in os.listdir("VBL_Images"):
        list_vals = masks[img_name]
        masked = np.asarray(list_vals)

        color_coding = {'background': (255, 255, 255), #white 
            'no-damage': (0,255,0), #green
            'minor-damage': (255,255,0), #yellow
            'major-damage': (255,165,0), #orange
            'destroyed': (255,0,0), #red
            'un-classified': (0,0,0) #black
            }  

        segmented_image = np.ones((1024, 1024, 3))  

        segmented_image[masked == 0] = color_coding['background']
        segmented_image[masked == 1] = color_coding['no-damage']
        segmented_image[masked == 2] = color_coding['minor-damage']
        segmented_image[masked == 3] = color_coding['major-damage']
        segmented_image[masked == 4] = color_coding['destroyed']

        from PIL import Image
        im = Image.fromarray(segmented_image.astype(np.uint8))
        im.save(f"Segmented_Images/segmented_{img_name}.png")

        img = cv2.imread(f"Segmented_Images/segmented_{img_name}.png")

        img1 = img // small_size
        img2 = img % small_size
        img1 = cv2.resize(img1.astype('uint8'), (small_size,small_size), interpolation=cv2.INTER_NEAREST)
        img2 = cv2.resize(img2.astype('uint8'), (small_size,small_size), interpolation=cv2.INTER_NEAREST)
        resized_image = img1.astype('uint16') * small_size + img2.astype('uint16')

        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        resized_image = np.array(resized_image)

        
        color_to_mask = {(255,255,255):0,(0,255,0):1,(255,255,0):2,(255,165,0):3,(255,0,0):4,(0,0,0):5}

        smaller_mask = np.ones((small_size,small_size))

        r = 0
        for row in resized_image:
            c=0
            for pixel in row:
                try:
                    smaller_mask[r][c] = int(color_to_mask[tuple(pixel)])
                except:
                    smaller_mask[r][c] = int(5)
                c+=1
            r+=1

        smaller_mask = smaller_mask.astype(int)
        x = smaller_mask.tolist()
        small_mask_values[f'{img_name}'] = x
        

    json_object = json.dumps(str(small_mask_values), separators=(',', ':'),indent = 4)

    with open(f"Train_Masks_{small_size}.json", "w") as outfile:
        outfile.write(json_object)

Create_Data(512)
