import os, sys, shutil
import json
import yaml
import random
import numpy as np
import csv
import re

aoi_dir_name = os.getenv('AOI_DIR_NAME')
assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"

current_dir = os.path.dirname(os.path.abspath(__file__))
idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
aoi_dir = current_dir[:idx]

# ========================== MY CONFIG =========================== #

# read path from yolov4.yaml
yml = yaml.safe_load(open('cfg/yolov4.yaml'))

# previous model
now_best_model = yml['yolov4_best_model']


# yolo config file for load model
project_name = yml['project_name']
input_dataset_path = yml['input_dataset_path']

Yolo_config_path = yml['YOLO_config_path']
Yolo_data_file  = os.path.join(Yolo_config_path, project_name+'.data')
Yolo_names_file = os.path.join(Yolo_config_path, project_name+'.names')
Yolo_cfg_file   = os.path.join(Yolo_config_path, project_name+'.cfg')
Yolo_train_file = os.path.join(Yolo_config_path, 'train.txt')
Yolo_valid_file = os.path.join(Yolo_config_path, 'valid.txt')



# yolo dataset path
Yolo_dataset_path = yml['YOLO_dataset_path']
Yolo_train_set_path = os.path.join(Yolo_dataset_path, 'train_data') 
Yolo_valid_set_path = os.path.join(Yolo_dataset_path, 'valid_data') 
Yolo_test_set_path  = os.path.join(Yolo_dataset_path, 'test_data') 

# training models folder
Yolo_weights_path = yml['YOLO_weight_path']


# input data detail
width = float(yml['width'])
height = float(yml['height'])
set_ratio = float(yml['set_ratio'])


# output valid.csv
test_GT_csv = os.path.join(Yolo_config_path, yml['test_GT_csv'])
valid_GT_csv = os.path.join(Yolo_config_path, yml['valid_GT_csv'])

# ================================================================ #

def write_inference_data(data,result_csv,method):
    defect_types_count_list = np.zeros(len(defects))
    # CSV
    with open(result_csv, method, newline='') as csvFile:
        #
        writer = csv.writer(csvFile)
        for d in data:
            defect_type = defects.index(d[1])
            defect_types_count_list[defect_type] += 1
            writer.writerow(d)
    return defect_types_count_list
    
def get_json_data(file_list):

    bbox_data = []
    for img_name,json_name in file_list:
    
        # read json array
        json_array = json.load(open (json_name))
        
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
    
            wrt_row = [img_name, item['label'], str(x), str(y), str(err_w), str(err_h)]
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

def generate_yolo_dataset(input_image_path, target_path, set_ratio = 0.8):
    #{"shapes": [{"label": "elh", "points": [[450,127],[489,168]],...},{"label": "excess_solder","points": [[452,299],[491,339]],...}],......}

    train_set =[]
    valid_set =[]
    defects = [] 

    image_list = get_file(input_image_path,".jpg")
    random.shuffle(image_list)
    for idx in range(len(image_list)):
        file_name = image_list[idx].split('.', 1 )[0]
        json_name = os.path.join(input_image_path, file_name+".json")
        json_array = json.load(open(json_name)) 
        for item in json_array['shapes']:
            if item['label'] not in defects :
                defects.append(item['label'])
    defects.sort()
    print(defects)
    
    for idx in range(len(image_list)):
        file_name = image_list[idx].split('.', 1 )[0]
        img_name = image_list[idx]
        json_name = os.path.join(input_image_path, file_name+".json")
        json_array = json.load(open(json_name)) 

        if float(idx)/float(len(image_list)) < set_ratio:
            set_path = Yolo_train_set_path
            if img_name not in train_set:
                train_set.append([img_name,json_name])
        else:
            set_path = Yolo_valid_set_path
            if img_name not in valid_set:
                valid_set.append([img_name,json_name])
        
        # generate yolo format txt
        fp = open(os.path.join(set_path, file_name+".txt"), "a")
        for item in json_array['shapes']:
            #if item['label'] not in defects : defects.append(item['label'])
            
            category_id = defects.index(item['label'])
            [bbox_l,bbox_t] = item['points'][0][:]
            [bbox_r,bbox_b] = item['points'][1][:]
            yolo_x = ((float(bbox_l)+float(bbox_r))/2) / width
            yolo_y = ((float(bbox_t)+float(bbox_b))/2) / height
            yolo_w = (abs(float(bbox_l)-float(bbox_r))) / width
            yolo_h = (abs(float(bbox_t)-float(bbox_b))) / height

            fp.write("%s %f %f %f %f\n" % (category_id, yolo_x, yolo_y, yolo_w, yolo_h))

        fp.close()
        
        # copy image to train set and valid set
        
        if not os.path.exists(os.path.join(set_path, img_name)):
            shutil.copyfile(os.path.join(input_image_path, img_name), os.path.join(set_path, img_name))


    '''
    for class_name in defects:
        now_dir = os.path.join(input_image_path,class_name)
        image_list = get_file(now_dir,"jpg")
        
        random.shuffle(image_list)

        for idx in range(len(image_list)):
            file_name = image_list[idx].split('.', 1 )[0]
            img_name = image_list[idx]
            json_name = os.path.join(now_dir, file_name+".json")
            json_array = json.load(open(json_name)) 

            if float(idx)/float(len(image_list)) < set_ratio:
                set_path = Yolo_train_set_path
                if img_name not in train_set:
                    train_set.append([img_name,json_name])
            else:
                set_path = Yolo_valid_set_path
                if img_name not in valid_set:
                    valid_set.append([img_name,json_name])
            
            # generate yolo format txt
            fp = open(os.path.join(set_path, file_name+".txt"), "a")
            for item in json_array['shapes']:
                category_id = defects.index(item['label'])
                [bbox_l,bbox_t] = item['points'][0][:]
                [bbox_r,bbox_b] = item['points'][1][:]
                yolo_x = ((float(bbox_l)+float(bbox_r))/2) / width
                yolo_y = ((float(bbox_t)+float(bbox_b))/2) / height
                yolo_w = (abs(float(bbox_l)-float(bbox_r))) / width
                yolo_h = (abs(float(bbox_t)-float(bbox_b))) / height

                fp.write("%s %f %f %f %f\n" % (category_id, yolo_x, yolo_y, yolo_w, yolo_h))

            fp.close()
            
            # copy image to train set and valid set
            
            if not os.path.exists(os.path.join(set_path, img_name)):
                shutil.copyfile(os.path.join(now_dir, img_name), os.path.join(set_path, img_name))
    '''

    return train_set, valid_set, defects


def replace_by_value(keyword, replace_value, file):

    for idx, line in enumerate(file): 
        if "#" in line : continue
        line = line.replace(" ", "")

        if keyword+"=" in line:
            file[idx] = keyword+"="+replace_value+"\n"
    
    return file



if __name__ == "__main__":

    # 1. check previous best training model is exists or not
    if now_best_model and os.path.exists(now_best_model):
        print("Use best model:",now_best_model)     
    else:
        now_best_model = yml['yolov4_model_file_path']
        print("Use yolov4 pre-trained model:", now_best_model)



    # 2. empty previous training model
    if os.path.exists(Yolo_weights_path):
        shutil.rmtree(os.path.abspath(Yolo_weights_path), ignore_errors=True)
    os.mkdir(Yolo_weights_path)



    # 3. prepare yolov4 dataset
    if os.path.exists(Yolo_train_set_path):
        print("Yolo_train_set_path: ",Yolo_train_set_path)
        shutil.rmtree(os.path.abspath(Yolo_train_set_path), ignore_errors=True)
    if os.path.exists(Yolo_valid_set_path):
        print("Yolo_valid_set_path: ",Yolo_valid_set_path)
        shutil.rmtree(os.path.abspath(Yolo_valid_set_path), ignore_errors=True)
    if os.path.exists(Yolo_dataset_path):
        print("Yolo_dataset_path: ",Yolo_dataset_path)
        shutil.rmtree(os.path.abspath(Yolo_dataset_path), ignore_errors=True)
    os.makedirs(Yolo_dataset_path)
    os.makedirs(Yolo_train_set_path)
    os.makedirs(Yolo_valid_set_path)



    # 4. generate yolov4 data file
    train_set,valid_set, defects= generate_yolo_dataset(input_dataset_path, Yolo_dataset_path, set_ratio)
    print("train_set size:" , len(train_set))
    print("valid_set size:" , len(valid_set))

    # 5. set yolo training config
    fp = open(Yolo_data_file, "w")
    fp.write("classes = %d\n" % (len(defects)))
    fp.write("train = %s\n" % (os.path.abspath(Yolo_train_file)))
    fp.write("valid = %s\n" % (os.path.abspath(Yolo_valid_file)))
    fp.write("names = %s\n" % (os.path.abspath(Yolo_names_file)))
    fp.write("backup = %s\n" % (os.path.abspath(Yolo_weights_path)))
    fp.write("eval = %s\n" % (project_name))
    fp.close()

    fp = open(Yolo_names_file, "w")
    for dd in defects:
        fp.write("%s\n" % (dd))
    fp.close()




    # 6. generate train.txt & valid.txt
    fp = open(Yolo_train_file, "w")
    for img in os.listdir(Yolo_train_set_path):
        abs_path = os.path.abspath(Yolo_train_set_path)
        if img.endswith(".jpg"):
            fp.write("%s\n" % os.path.join(abs_path, img))
    fp.close()

    fp = open(Yolo_valid_file, "w")
    for img in os.listdir(Yolo_valid_set_path):
        abs_path = os.path.abspath(Yolo_valid_set_path)
        if img.endswith(".jpg"):
            fp.write("%s\n" % os.path.join(abs_path, img))
    fp.close()

    # 7. prepare yolo cfg    
    classes = len(defects)

    shutil.copyfile(os.path.join(Yolo_config_path, yml['select_cfg']), Yolo_cfg_file)
    with open(Yolo_cfg_file, 'r') as f:
        cfg = f.readlines()
        

    cfg = replace_by_value('batch', yml['batch'], cfg)
    cfg = replace_by_value('subdivisions', yml['subdivisions'], cfg)
    cfg = replace_by_value('width', yml['width'], cfg)
    cfg = replace_by_value('height', yml['height'], cfg)
    cfg = replace_by_value('channels', yml['channels'], cfg)
    cfg = replace_by_value('learning_rate', yml['learning_rate'], cfg)


    if yml['max_batches']:
        cfg = replace_by_value('max_batches', yml['max_batches'], cfg)
        cfg = replace_by_value('steps', "{},{}".format( int(int(yml['max_batches'])*0.8), int(int(yml['max_batches'])*0.9)), cfg)

    else:
        cfg = replace_by_value('max_batches', int(classes*2000), cfg)
        cfg = replace_by_value('steps', "{},{}".format( int(classes*2000*0.8), int(classes*2000*0.9)), cfg)

    # change classes and filter
    for idx, line in enumerate(cfg): 
        if "#" in line : continue
        line = line.replace(" ", "")

        if "classes=" in line:
            cfg[idx] = "classes="+str(classes)+"\n"
            for ii in range(idx, 0, -1):
                cfg[ii] = cfg[ii].replace(" ", "")
                if "filters=" in cfg[ii]: 
                    cfg[ii] = "filters="+str(int((classes + 5)*3))+"\n"
                    break
    
    
    with open(Yolo_cfg_file, 'w') as f:
        f.writelines(cfg)


    # 8.prepare GT
    valid_GT_data = get_json_data(valid_set)
    valid_defect_count = write_inference_data(valid_GT_data,valid_GT_csv,'w')
    print("valid_defect_count: ",valid_defect_count)


