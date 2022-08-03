import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import numpy
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon


def Create_Data():
    dmg_type = {'background': 0, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4,
                    'un-classified': 5}

    def get_buildings(annotation_path):
        annotation = pd.read_json(annotation_path)
        buildings = dict()
        for b in annotation['features']['xy']:
            buid = b['properties']['uid']
            poly = Polygon(wkt.loads(b['wkt']))
            buildings[buid] = {'poly': list(poly.exterior.coords),
                            'subtype': b['properties']['subtype']}
        return buildings

    def make_mask_img(**kwargs):
        width=1024
        height=1024
        builings=kwargs
        mask_img = np.zeros([height, width], dtype=np.uint8)
        for dmg in dmg_type:
            polys_dmg = [np.array(builings[p]['poly']).round().astype(np.int32).reshape(-1, 1, 2)
                            for p in builings if builings[p]['subtype'] == dmg]
            cv2.fillPoly(mask_img, polys_dmg, [dmg_type[dmg]])

        return mask_img

    annotations = './VisionBeyondLimits/Labels/'

    mask_values = dict()

    for annotation_path in os.listdir(annotations):
        path = os.path.join(annotations,annotation_path)
        annotation = pd.read_json(path)
        img_name = annotation['metadata']['img_name']
        buildings = get_buildings(path)
        masked = make_mask_img(**buildings) #This is a 1024*1024 array
        mask_values[f'{img_name}'] = masked.tolist()
        
    

    json_object = json.dumps(str(mask_values), separators=(',', ':'),indent = 4)

    with open("Train_Masks_1024.json", "w") as outfile:
        outfile.write(json_object)


Create_Data()







