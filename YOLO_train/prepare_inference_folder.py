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
if os.path.exists(os.path.join(YOLO_inference_path, 'test')):
    shutil.rmtree(os.path.abspath(os.path.join(YOLO_inference_path, 'test')), ignore_errors=True)
os.mkdir(os.path.join(YOLO_inference_path, 'test'))
os.mkdir(os.path.join(YOLO_inference_path, 'test', project_name))

