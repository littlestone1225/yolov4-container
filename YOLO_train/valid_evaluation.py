import csv, json
import os,shutil
import numpy as np
import logging
import yaml


# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))

project_name = yml['project_name']



Root_data_path =  os.path.join(yml['YOLO_inference_path'], 'valid')
inference_csv = os.path.join(Root_data_path , yml['valid_eval_csv'])

Yolo_config_path = yml['YOLO_config_path']
Yolo_names_file = os.path.join(Yolo_config_path, project_name+'.names')

Yolo_result_csv =  yml['yolo_valid_csv']
VALID_GT_csv = os.path.join(Yolo_config_path , yml['valid_GT_csv'])


crop_size = yml['window_size']
iou_threshold = 0.1
score_threshold = 0.01


with open(Yolo_names_file) as f:
    defects = [line.rstrip('\n') for line in f]

print(defects)
#######################################################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def empty_all_statistics():
    global label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict
    global truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict
    global infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict

    """
    Ground Truth Statistics
    1. truth_dict                   : all of the ground truth
    2. truth_tp_dict                : those ground truth which are detected by detector with correct category (the same as infer_tp_dict)
    3. truth_tp_wrong_category_dict : those ground truth which are detected by detector with wrong category
    4. truth_fn_dict                : those ground truth which are not detected by detector

    [Note] truth_dict = truth_tp_dict + truth_tp_wrong_category_dict + truth_fn_dict
           the sum of truth_tp_wrong_category_dict = the sum of infer_tp_wrong_category_dict
           Combine 'empty', 'appearance_less' and 'appearance_hole' of ground truth label statistics into 'appearance_less' of ground truth statistics
    """
    truth_dict                   = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    truth_tp_dict                = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    truth_tp_wrong_category_dict = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    truth_fn_dict                = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}

    """
    Inference Result Statistics
    1. infer_dict                   : all of the detected object
    2. infer_tp_dict                : those detected objects which match the ground truth with correct category (the same as truth_tp_dict)
    3. infer_tp_wrong_category_dict : those detected objects which match the ground truth with wrong category
    4. infer_fp_dict                : those detected objects which does not match the ground truth

    [Note] infer_dict = infer_tp_dict + infer_tp_wrong_category_dict + infer_fp_dict
           the sum of truth_tp_wrong_category_dict = the sum of infer_tp_wrong_category_dict
           Combine 'empty', 'appearance_less' and 'appearance_hole' of ground truth label statistics into 'appearance_less' of inference result statistics
    """
    infer_dict                   = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    infer_tp_dict                = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    infer_tp_wrong_category_dict = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}
    infer_fp_dict                = {'bridge': 0, 'elh': 0, 'excess_solder': 0, 'appearance': 0}

def dict_to_str(dict_obj):
    """
    Conver dict to formatted string

    Args:
        dict_obj (dict): dict.
    Returns:
        str_obj (str): formatted string.
    """
    str_obj = ""
    for key in dict_obj:
        str_add = "%s = %6d" % (key, dict_obj[key])
        str_obj = str_obj + str_add + "; "
    return str_obj


def get_normal_recall(truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict):
    TP = 0
    FN = 0
    for key in truth_dict:
        TP = TP + truth_tp_dict[key] + truth_tp_wrong_category_dict[key]
        FN = FN + truth_fn_dict[key]
    recall = round(TP / (TP + FN), 3)
    return recall

def get_normal_precision(infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict):
    TP = 0
    FP = 0
    for key in infer_dict:
        TP = TP + infer_tp_dict[key] + infer_tp_wrong_category_dict[key]
        FP = FP + infer_fp_dict[key]
    precision = round(TP / (TP + FP), 3)
    return precision

def output_result(result_dir):
    print(result_dir)
    print(Yolo_result_csv)
    logging.info("score_threshold              : {}".format(score_threshold))
    #logging.info("min_score                    : {}".format(min_score))

    logging.info("truth_dict                   : {}".format(dict_to_str(truth_dict)))
    logging.info("truth_tp_dict                : {}".format(dict_to_str(truth_tp_dict)))
    logging.info("truth_tp_wrong_category_dict : {}".format(dict_to_str(truth_tp_wrong_category_dict)))
    logging.info("truth_fn_dict                : {}".format(dict_to_str(truth_fn_dict)))
    logging.info("")
    logging.info("infer_dict                   : {}".format(dict_to_str(infer_dict)))
    logging.info("infer_tp_dict                : {}".format(dict_to_str(infer_tp_dict)))
    logging.info("infer_tp_wrong_category_dict : {}".format(dict_to_str(infer_tp_wrong_category_dict)))
    logging.info("infer_fp_dict                : {}".format(dict_to_str(infer_fp_dict)))
    logging.info("")
    
    normal_recall = get_normal_recall(truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict)
    logging.info("normal recall                : {}".format(normal_recall))
    normal_precision = get_normal_precision(infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict)
    logging.info("normal precision             : {}".format(normal_precision))
    
    logging.info("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
    
    write_row = [result_dir.split("_")[-1],Yolo_result_csv,truth_tp_dict,truth_fn_dict,normal_recall,normal_precision]

    return write_row
    


    


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

def compare_GT_YOLO(result_dir):
    global label_dict, label_tp_dict, label_tp_wrong_category_dict, label_fn_dict
    global truth_dict, truth_tp_dict, truth_tp_wrong_category_dict, truth_fn_dict
    global infer_dict, infer_tp_dict, infer_tp_wrong_category_dict, infer_fp_dict

    #read file
    with open(VALID_GT_csv, newline='') as f:
        reader = csv.reader(f)
        GT_data = list(reader)
    
    with open(os.path.join(Root_data_path, result_dir, Yolo_result_csv), newline='') as csvfile:
        reader = csv.reader(csvfile)
        YOLO_data = list(reader)
    print(len(GT_data))
    print(len(YOLO_data))
    GT_res = np.zeros(len(GT_data)+1)
    yolo_res = np.zeros(len(YOLO_data)+1)

    # truth 
    i = 0
    for row in GT_data:
        i = i + 1

        img_name = row[0].split("=")[0]
        GT_type = defects.index(row[1])
        GT_name = row[1]
            
        truth_dict[GT_name] += 1

        x = int(row[2])
        y = int(row[3])
        err_w = int(row[4])
        err_h = int(row[5])
        #print(row)

        gt_box = [x, y, x+err_w, y+err_h]
        #print(i,",",img_name)
        pos_idx_list = index_2d(YOLO_data, img_name)
        #print(pos_idx_list)
        
        
        iou_list = []
        k = 0
        # First, choose the highest overlap area ratio. Second, choose the highest confidence score
        for pos in pos_idx_list:
            k+=1
            if float(YOLO_data[pos[0]][6]) < score_threshold:
                iou_list.append(0.0)
                continue
            
            pos_l, pos_t, pos_w, pos_h = YOLO_data[pos[0]][2:6]

            pred_box = [int(pos_l), int(pos_t), int(pos_l)+min(crop_size, int(pos_w)), int(pos_t)+min(crop_size, int(pos_h))]

            
            iou = get_iou(pred_box, gt_box) 
            #print(k,pos[0],float(YOLO_data[pos[0]][6]),score_threshold,iou)
            iou_list.append(iou)
            if iou > iou_threshold:
                yolo_res[pos[0]] = 1


        # GT
        if(len(iou_list)<1):
            truth_fn_dict[GT_name] += 1
            continue

        max_iou = max(iou_list)
        if max_iou < iou_threshold: # FN
            truth_fn_dict[GT_name] += 1
        else:
            max_iou_pos = [i for i, iou in enumerate(iou_list) if iou == max_iou]
            max_score = 0

            for pos in max_iou_pos:
                if float(YOLO_data[pos_idx_list[pos][0]][6]) < score_threshold:
                    continue

                if max_score< float(YOLO_data[pos_idx_list[pos][0]][6]):
                    max_score = float(YOLO_data[pos_idx_list[pos][0]][6])
                    select = pos_idx_list[pos][0]

            YOLO_name = YOLO_data[select][1]
            YOLO_type = defects.index(YOLO_data[select][1])
            yolo_res[select] = GT_type
            
            if GT_name == YOLO_name: #TP
                truth_tp_dict[GT_name] += 1
                infer_tp_dict[YOLO_name] += 1
                GT_res[i] = 1
            else:                   #TP wrong category
                truth_tp_wrong_category_dict[GT_name] += 1
                infer_tp_wrong_category_dict[YOLO_name] += 1
                GT_res[i] = YOLO_type
            
            #if i == 100:break
        
        
    #YOLO
    j = 0
    for row in YOLO_data:
        
        if float(row[6]) >= score_threshold:
            infer_dict[row[1]] += 1
            YOLO_type = defects.index(row[1])
            if yolo_res[j] ==0:
                infer_fp_dict[row[1]] += 1
        j += 1
        
        
def write_inference_data(data,result_csv,method):
    # CSV
    with open(result_csv, method, newline='') as csvFile:
        #
        writer = csv.writer(csvFile)
        for d in data:
            writer.writerow(d)

    return
    
    
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



if __name__ == '__main__':

    i=0
    #load all test board images
    result_list = []
    for directory in os.listdir(Root_data_path):
        i+=1
        #if i>2:break
        if os.path.isdir(os.path.join(Root_data_path, directory)):
            result_list.append(directory)
    result_list.sort()
    
    
    inference_data = []
    for res_dir in result_list:
        # start evaluation
        empty_all_statistics()
        
        compare_GT_YOLO(res_dir)
        
        inference_dict = output_result(res_dir)
        #print(inference_dict)
        inference_data.append(inference_dict)
        
 
    inference_data.sort()  
    write_inference_data(inference_data,inference_csv,"w")

    
    