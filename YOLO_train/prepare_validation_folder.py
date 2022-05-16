import yaml
import os, shutil
from yolov4_pre_process import replace_by_value
# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))

# inference folder
YOLO_inference_path = yml['YOLO_inference_path']
project_name = yml['project_name']

Yolo_config_path = yml['YOLO_config_path']
Yolo_cfg_file   = os.path.join(Yolo_config_path, project_name+'.cfg')

# ================================================================ #

# remove before result
if os.path.exists(YOLO_inference_path):
    shutil.rmtree(os.path.abspath(YOLO_inference_path), ignore_errors=True)
os.mkdir(YOLO_inference_path)
os.mkdir(os.path.join(YOLO_inference_path, 'valid'))
os.mkdir(os.path.join(YOLO_inference_path, 'valid', project_name))


# 9. setting anchor 
with open(Yolo_cfg_file, 'r') as f:
    cfg = f.readlines()

with open("anchors.txt", 'r') as f:
    anchor = f.read()    
cfg = replace_by_value("anchors", "{}".format(anchor.split("=")[-1]), cfg)


# write config file
with open(Yolo_cfg_file, 'w') as f:
    f.writelines(cfg)