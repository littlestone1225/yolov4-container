import csv, json
import os,shutil
import numpy as np
import logging
import yaml
from valid import *

# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))

project_name = yml['project_name']

TEST_data_path = os.path.expanduser(yml['TEST_data_path'])  

# inference folder
YOLO_inference_path = yml['YOLO_inference_path']
Root_data_path =  os.path.join(yml['YOLO_inference_path'], 'test', project_name)

if yml['yolov4_best_model'] == None:
    YOLO_weight_path = yml['pretrained']
else:
    YOLO_weight_path = yml['yolov4_best_model']

Yolo_config_path = yml['YOLO_config_path']
Yolo_data_file  = os.path.join(Yolo_config_path, project_name+'.data')
Yolo_names_file = os.path.join(Yolo_config_path, project_name+'.names')
Yolo_cfg_file   = os.path.join(Yolo_config_path, project_name+'.cfg')

Yolo_result_csv =  yml['yolo_valid_csv']



# classes
with open(Yolo_names_file) as f:
    defects = [line.rstrip('\n') for line in f]
print(defects)


# detail
width = int(yml['width'])
height = int(yml['height'])
Margin = 100

NMS_flag = int(yml['NMS_flag'])
NMS_Iou_threshold = float(yml['NMS_Iou_threshold'])
Edge_limit = int(yml['Edge_limit'])

Batch_size = int(yml['inference_Batch_size'])

# inference
Score_threshold = float(yml['Score_threshold'])

#######################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)
    
    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    
    return iou

def index_2d(data, search):
    pos_idx = []
    for i, e in enumerate(data):
        try:
            pos_idx.append([i,e.index(search)])
            #return i, e.index(search)
        except ValueError:
            pass
    return pos_idx


    
def get_json_data(file_list,label_path):

    bbox_data = []
    for img in file_list:
    
        # read json array
        json_array = json.load(open(os.path.join(label_path, img+'.json')))  
        
        g = 0
        for item in json_array["shapes"]:
            g = g+1
            
            #print(item)
    
            defect_type = defects.index(item['label'])
            x = int(item['points'][0][0])
            y = int(item['points'][0][1])
            x2 = int(item['points'][1][0])
            y2 = int(item['points'][1][1])
    
            err_w = x2 - x
            err_h = y2 - y
    
            wrt_row = [img+'.jpg', item['label'], str(x), str(y), str(err_w), str(err_h)]
            bbox_data.append(wrt_row)
            #print(wrt_row)


    return bbox_data

def get_file(root_folder,file_type):
    file_list = []
    for file_name in os.listdir(root_folder):
        if file_name.endswith(file_type):
            file_list.append(file_name)
    file_list.sort()
    
    return file_list


if __name__ == '__main__':
    print(YOLO_weight_path)
    now_weights = YOLO_weight_path.split('/')[-1].split('.')[0]
    now_Result_folder = os.path.join(Root_data_path, now_weights)
    print("now processing :", now_Result_folder)
    Yolo_result_label_json_dir = os.path.join(now_Result_folder, 'label_json_dir')
    
    make_directory(now_Result_folder , 0 )
    make_directory(Yolo_result_label_json_dir, 0)

    # load model
    network, class_names, class_colors = load_darknet(Yolo_cfg_file,Yolo_data_file, \
                                                            YOLO_weight_path, Batch_size)

    #load all test images
    images_list = get_file(TEST_data_path,'.jpg')
    for img_cnt, img in enumerate(images_list):

        # read image for board
        I = cv2.imread(os.path.join(TEST_data_path, img))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = I.astype(np.uint8)
        print("now process:",img_cnt,img)

        # crop big img to patches
        crop_rect_list, crop_image_list = spilt_patches(I, width, height, Margin)
        print("total patches: ",len(crop_image_list))


        # detection	
        if Batch_size == 1 :
            yolo_data = image_detection(crop_image_list, crop_rect_list, img, network, \
                                                class_names, class_colors, Score_threshold, NMS_flag, \
                                                Edge_limit,NMS_Iou_threshold)
        else : 
            yolo_data = batch_detection(crop_image_list, crop_rect_list, img, network, \
                                                class_names, class_colors, Score_threshold, NMS_flag, \
                                                Edge_limit,NMS_Iou_threshold, .5, .45, Batch_size)

        # change format
        csv_to_json(yolo_data, TEST_data_path, Yolo_result_label_json_dir, "xmin_ymin_w_h", True)

        # record detect bbox
        write_data_to_YOLO_csv_100(yolo_data,Yolo_result_csv,now_Result_folder,"a")
    
    # free GPU memory
    free_darknet(network)
    