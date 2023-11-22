
import argparse
from PIL import Image
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import shutil


img2idx_json="/home/tj/QCR_PubMedCLIPs_fusion/data/data_slake/train.json"
img_path="/home/tj/QCR_PubMedCLIPs_fusion/data/data_slake/imgs/"
with open(img2idx_json) as f:
    img2idx = json.load(f)
print("over")
list_brain = [ "Brain_Tissue", "Brain_Face", "Brain"]
list_chest = [ "Lung", "Chest_heart", "Chest_lung","Chest_mediastinal"]

for entry in img2idx:
    image_name=entry['img_name']
    new_name,_=image_name.split('/')
    new_name=new_name+'.jpg'
    if entry['location']== "Abdomen":
        img_save_path="/home/tj/slake_class/ab/"
    elif entry['location'] in list_brain:
        img_save_path="/home/tj/slake_class/brain/"
    elif entry['location'] in list_chest:
        img_save_path="/home/tj/slake_class/chest/"
    elif entry['location'] == "Neck":
        img_save_path="/home/tj/slake_class/neck/"
    else:
        img_save_path="/home/tj/slake_class/pel/"
    new_name=img_save_path+new_name
    source_path=img_path+image_name
    shutil.copy(source_path, img_save_path)
    # shutil.copyfile(os.path.join(source_path,new_name),os.path.join(source_path,new_name))

