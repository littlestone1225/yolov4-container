import yaml
import os, shutil
# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))

# inference folder
YOLO_inference_path = yml['YOLO_inference_path']
project_name = yml['project_name']

# ================================================================ #

# remove before result
if os.path.exists(YOLO_inference_path):
    shutil.rmtree(os.path.abspath(YOLO_inference_path), ignore_errors=True)
os.mkdir(YOLO_inference_path)
os.mkdir(os.path.join(YOLO_inference_path, 'valid'))
os.mkdir(os.path.join(YOLO_inference_path, 'valid', project_name))
