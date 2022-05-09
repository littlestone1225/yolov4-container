import os, shutil
import matplotlib.pyplot as plt
import csv
import json
from collections import OrderedDict
import time
import numpy as np
import darknet
import cv2
import math
import argparse
import yaml




def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--batch_size", default=Batch_size, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default=yml['yolov4_model_file_path'],
                        help="yolo weights path")
    return parser.parse_args()


def spilt_patches(img, width,height, margin):

    h_stride = height-margin
    w_stride = width-margin
    h_step = height
    w_step = width
    sh = list(img.shape)
    if img.shape[0] == height and img.shape[1] == width:
        print("size == cropped")
        splitted_pos = [[0,0,width,height]]
        splitted_img = [img]
        return splitted_pos, splitted_img
    #sh[0], sh[1] = sh[0] + margin * 2, sh[1] + margin * 2
    nrows, ncols = math.ceil(img.shape[0] / h_stride), math.ceil(img.shape[1] / w_stride)
    sh[0] = nrows * h_stride + margin
    sh[1] = ncols * w_stride + margin
    color = (0,0,0)
    img_ = np.full(sh, color, dtype=np.uint8)
    img_[0:img.shape[0], 0:img.shape[1],:] = img
    
    splitted_img = []
    splitted_pos = []
    for j in range(nrows):
        for i in range(ncols):
            h_start = j*stride
            v_start = i*stride
            cropped = img_[ h_start:h_start+h_step ,v_start:v_start+w_step,:]

            #output
            splitted_pos.append([v_start, h_start, v_start+w_step, h_start+h_step])
            splitted_img.append(cropped)
    
    splitted_pos = np.array(splitted_pos,dtype='int32')
    splitted_img = np.array(splitted_img,dtype='uint8')
    return splitted_pos, splitted_img

def make_directory(folder_path,remove_anyway):
	if remove_anyway:
		# remove folder anyway
		if os.path.exists(folder_path): shutil.rmtree(folder_path, ignore_errors=True)
		if not os.path.exists(folder_path): os.mkdir(folder_path)
	elif not os.path.exists(folder_path):
		os.mkdir(folder_path)

	return

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

def csv_to_json(label_array, image_dir, label_json_dir, coord_type="xmin_ymin_w_h", store_score=False):
    """
    Convert csv format to labelme json format.
    Args:
        label_array (list[list] or np.ndarray): annotations in [[file_name, error_type, xmin, ymin, xmax, ymax, score], ...] format.
                                                (score is optional and is controlled by store_score)
        image_dir (str): directory of image file (file_name = xxx.jpg).
        label_json_dir (str): directory in which json file is stored
        coord_type (str): "xmin_ymin_w_h" or "xmin_ymin_xmax_ymax"
        store_score (boolean): determine if you want to store score in json file or not
    """
	
    json_dict = OrderedDict()

    if len(label_array) > 0:
        for i, label in enumerate(label_array):
            file_name  = label[0].split("\n")[0]
            error_type = label[1]
            if coord_type == "xmin_ymin_w_h":
                xmin, ymin, w, h = [int(i) for i in label[2:6]]
                xmax = xmin + w
                ymax = ymin + h
            elif coord_type == "xmin_ymin_xmax_ymax":
                xmin, ymin, xmax, ymax = [int(i) for i in label[2:6]]
            else:
                assert False, 'coord_type should be either xmin_ymin_w_h or xmin_ymin_xmax_ymax'
            
            if i==0:
                json_dict["version"] = "4.5.6"
                json_dict["flags"] = dict()
                json_dict["shapes"] = list()
                json_dict["imagePath"] = file_name
                json_dict["imageData"] = None

                image_file_path = os.path.join(image_dir, file_name)
               
                if os.path.isfile(image_file_path):
                    image = plt.imread(image_file_path)
                    json_dict["imageHeight"] = image.shape[0]
                    json_dict["imageWidth"] = image.shape[1]
                else:
                    #logging.warning("{} does not exist".format(image_file_path))
                    return

            shapes = OrderedDict()
            shapes["label"] = error_type
            shapes["points"] = [[xmin, ymin], [xmax, ymax]]
            shapes["group_id"] = None
            shapes["shape_type"] = "rectangle"
            shapes["flags"] = dict()
            if store_score and len(label) >= 7:
                score = float(label[6])
                shapes["score"] = score
            json_dict["shapes"].append(shapes)
            
        
        json_file_name = os.path.splitext(file_name)[0] + '.json'
        json_file_path = os.path.join(label_json_dir, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))

def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_resized = cv2.resize(image, (width, height),interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)

def batch_detection(patches_list, patches_position, filename, network, class_names, class_colors, \
                    thresh, nms_flag,edge_limit,iou_threshold ,hier_thresh, nms, batch_size):

    height = patches_list[0].shape[0]
    width = patches_list[0].shape[1]
    images_list = patches_list.copy()
    i = 0
    YOLO_detections = []

    images_cnt = len(images_list)
    print("images_cnt:",images_cnt)
    while images_cnt:
        patches = []
        now_patch_idx = i
        now_size = min(images_cnt,batch_size)
        for idx in range(now_size):
            patches.append(images_list[i])
            i += 1
            images_cnt -= 1
            
        image_height, image_width, _ = check_batch_shape(patches, now_size)
        darknet_images = prepare_batch(patches, network)

        batch_detections = darknet.network_predict_batch(network, darknet_images, now_size, width, height, thresh, hier_thresh, None, 0, 0)
        i = now_patch_idx
        for idx in range(now_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                darknet.do_nms_obj(detections, num, len(class_names), nms)
            predictions = darknet.remove_negatives(detections, class_names, num)

            YOLO_detections += detection_process(predictions,filename,patches_position[i],edge_limit,width,height)
            i += 1
                
        darknet.free_batch_detections(batch_detections, now_size)
       
    # NMS
    if nms_flag:
        YOLO_detections = bbox_nms(YOLO_detections,iou_threshold)

    if(len(YOLO_detections))<=0: 
        print("error img: ",filename)
    
    
    return YOLO_detections


def image_detection(patches_list, patches_position, filename, network, class_names, class_colors, thresh,nms_flag,edge_limit,iou_threshold):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    YOLO_detections = []
    detect_images_list = []

    i = 0
    for patch in patches_list:
        
        patch_resized = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, patch_resized.tobytes())

        #detect patch
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

        #detection result process
        YOLO_detections += detection_process(detections,filename,patches_position[i],edge_limit, width, height)
        
        i+=1
    # NMS
    if nms_flag:
        YOLO_detections = bbox_nms(YOLO_detections,iou_threshold)
    
    darknet.free_image(darknet_image)
    return  YOLO_detections

def detection_process(detections,filename,position,edge_limit,width, height):
    selected_detections = []
    w_edge_limit_max = width-edge_limit
    w_edge_limit_min = edge_limit
    h_edge_limit_max = height-edge_limit
    h_edge_limit_min = edge_limit

    # data = [type, score, [l,t,r,b]]
    for data in detections:
        if data[2][0] < w_edge_limit_min or data[2][1] < h_edge_limit_min or data[2][0] > w_edge_limit_max or data[2][1] > h_edge_limit_max :continue
        if data[2][2] > width or data[2][3] > height: continue

        new_w = int(data[2][2])
        new_h = int(data[2][3])
        new_l = int(data[2][0] - (new_w/2)) + position[0]
        new_t = int(data[2][1] - (new_h/2)) + position[1]
        confidence = str(round(data[1],4))

        new_data = [filename,data[0],new_l,new_t,new_w,new_h,confidence]
        selected_detections.append(new_data)

    return selected_detections

def bbox_nms(detections,iou_threshold):
    nms_detections = []
    
    # compare bbox
    for j in range(len(detections)):
        l, t, w, h = detections[j][2:6]
        l = int(l)
        t = int(t)
        w = int(w)
        h = int(h)                
        first_box = [l, t, l+w, t+h]
        first_score = float(detections[j][6])
        wrt_flag = True

        for k in range(len(detections)):
            if j == k: continue
            pos_l, pos_t, pos_w, pos_h = detections[k][2:6]
            pos_l = int(pos_l)
            pos_t = int(pos_t)
            pos_w = int(pos_w)
            pos_h = int(pos_h)                
            second_box = [pos_l, pos_t, pos_l+pos_w, pos_t+pos_h]
            second_score = float(detections[k][6])

            iou = get_iou(first_box, second_box)
            #print(j,k,iou)
            if iou < iou_threshold: continue
            if first_score < second_score: 
                wrt_flag = False
                continue
            if first_score == second_score and j<k:
                wrt_flag = False
                continue
        if wrt_flag:
            wrt_row = detections[j]
            nms_detections.append(wrt_row)
            #print(wrt_row)

    return nms_detections


def write_data_to_YOLO_csv_100( yolo_data,yolo_result_csv,result_folder,method):
    # CSV
    with open(os.path.join(result_folder, yolo_result_csv), method, newline='') as csvFile:
        #
        writer = csv.writer(csvFile)
        for data in yolo_data:
            writer.writerow(data)
    return

def load_darknet(Yolo_cfg_file, Yolo_data_file, Yolo_weights_file, Batch_size):

    network, class_names, class_colors = darknet.load_network(
        Yolo_cfg_file,
        Yolo_data_file,
        Yolo_weights_file,
        Batch_size
    )

    return network, class_names, class_colors


def free_darknet(net):
    darknet.free_network_ptr(net)
    return

if __name__ == '__main__':
    # ========================== MY CONFIG =========================== #

    # read path from yolov4.yaml
    yml = yaml.safe_load(open('cfg/yolov4.yaml'))

    VAL_data_path = os.path.expanduser(yml['VAL_data_path'])  



    # yolo config file for load model
    Yolo_weights_path = yml['YOLO_weight_path']

    project_name = yml['project_name']
    input_dataset_path = yml['input_dataset_path']

    Yolo_config_path = yml['YOLO_config_path']
    Yolo_data_file  = os.path.join(Yolo_config_path, project_name+'.data')
    Yolo_names_file = os.path.join(Yolo_config_path, project_name+'.names')
    Yolo_cfg_file   = os.path.join(Yolo_config_path, project_name+'.cfg')

    Yolo_result_csv = yml['yolo_valid_csv']

    # inference folder
    YOLO_inference_path = yml['YOLO_inference_path']


    # detail
    width = int(yml['width'])
    height = int(yml['height'])
    Margin = 100


    Score_threshold = yml['height']

    NMS_flag = True
    Iou_threshold = 0.75
    Edge_limit = 20 #pixels

    Batch_size = 1

    #######################################################################

    args = parser()
    make_directory(YOLO_inference_path, 0 )
    make_directory(os.path.join(YOLO_inference_path, 'valid'), 0 )
    
    Yolo_weights_file = args.weights
    now_weights = Yolo_weights_file.split('/')[-1].split('.')[0]
    
    now_Result_folder = os.path.join(YOLO_inference_path, 'valid',now_weights)
    print("now processing :", now_Result_folder)
    Yolo_result_label_json_dir = os.path.join(now_Result_folder, 'label_json_dir')
    
    make_directory(now_Result_folder , 0 )
    make_directory(Yolo_result_label_json_dir, 0)
    
    # load model
    network, class_names, class_colors = darknet.load_network(
        Yolo_cfg_file,
        Yolo_data_file,
        Yolo_weights_file,
        Batch_size
    )

    #load all valid board images
    images_list = []
    for img in os.listdir(VAL_data_path):
        if img.endswith(".jpg"):
            images_list.append(img)
    images_list.sort()

    max_image_name = ""
    max_image_time = 0
    min_image_name = ""
    min_image_time = 30

    img_cnt = 0

    # detection for all test board images
    for img in images_list:
        img_cnt +=1
        #if img_cnt >2: break
        # read big image for board
        I = cv2.imread(os.path.join(VAL_data_path, img))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = I.astype(np.uint8)
        print("now process:",img_cnt,img)

        
        # crop big img to patches
        crop_rect_list, crop_image_list = spilt_patches(I, width, height, Margin)
        print("total patches: ",len(crop_image_list))
        
        # start detect time
        start = time.time()

        # detection	
        if Batch_size == 1 :
            yolo_data = image_detection(crop_image_list, crop_rect_list, img, network, class_names, class_colors, \
                                        Score_threshold,NMS_flag,Edge_limit,Iou_threshold)
        else : 
            yolo_data = batch_detection(crop_image_list, crop_rect_list, img, network, class_names, class_colors, \
                                        Score_threshold, NMS_flag,Edge_limit,Iou_threshold ,.5, .45, Batch_size)

        detect = time.time()

        # generate json
        csv_to_json(yolo_data, VAL_data_path, Yolo_result_label_json_dir, "xmin_ymin_w_h", True)

        # end detect time
        end = time.time()
        
        # write to csv
        write_data_to_YOLO_csv_100(yolo_data,Yolo_result_csv,now_Result_folder,"a")
        
        
        #
        detect_time = detect-start
        process_time = end - start
        print("bbox count: ",len(yolo_data))
        print("detect time : %f s" % (detect_time))
        print("each big img processing time : %f s" % (process_time))
        print("*" * 100)

        
        if process_time >= max_image_time:
            max_image_time = process_time
            max_image_name = img
        
        if process_time <= min_image_time:
            min_image_time = process_time
            min_image_name = img
        

    print("range of process_time: ", max_image_time," ~ ", min_image_time, " s")
    print("max_image: ", max_image_name)
    print("min_image: ", min_image_name)
      
