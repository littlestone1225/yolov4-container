import os, sys, shutil
import json
import yaml
import random
import numpy as np
import csv

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))


# yolo config file for load model
project_name = yml['project_name']

Yolo_config_path = yml['YOLO_config_path']
Yolo_cfg_file   = os.path.join(Yolo_config_path, project_name+'.cfg')

# ================================================================ #



if __name__ == "__main__":


    # 9. setting anchor 
    with open("anchors.txt", 'r') as f:
        anchor = f.read()
    print(anchor)

    with open(Yolo_cfg_file, 'r') as f:
        cfg = f.read()
        
    cfg = cfg.replace("anchors =  30, 30,  38, 39,  35, 60,  61, 35,  48, 49,  56, 62,  78, 48,  62, 78,  97, 83", "anchors={}".format(anchor))


    with open(Yolo_cfg_file, 'w') as f:
        f.write(cfg)



