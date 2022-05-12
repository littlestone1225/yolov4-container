#!/usr/bin/env python3
import os
import yaml
from collections import OrderedDict
from subprocess import check_output
from filelock import FileLock

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def read_config_yaml(config_file):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    if os.path.isfile(config_file):
        with lock:
            with open(config_file) as file:
                # config_dict = yaml.load(file, Loader=yaml.Loader)
                config_dict = ordered_load(file, yaml.SafeLoader)
                if config_dict==None:
                    config_dict = OrderedDict()
    else:
        config_dict = OrderedDict()
    return config_dict

def write_config_yaml(config_file, write_dict):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    config_dict = read_config_yaml(config_file)
    for key, value in write_dict.items():
        config_dict[key] = value

    with lock:
        with open(config_file, 'w') as file:
            # yaml.dump(config_dict, file, default_flow_style=False)
            ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def write_config_yaml_with_key_value(config_file, key, value):
    config_dict = read_config_yaml(config_file)
    config_dict[key] = value

    with open(config_file, 'w') as file:
        # yaml.dump(config_dict, file, default_flow_style=False)
        ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def print_config_yaml(config_file):
    config_dict = read_config_yaml(config_file)
    print(dict(config_dict))

def check_env(ori_value,ENV_NAME):
    ENV_VALUE = (ori_value,os.getenv(ENV_NAME))[os.getenv(ENV_NAME)!=None]
    return ENV_VALUE

if __name__ == "__main__":
    aoi_dir_name = os.getenv('AOI_DIR_NAME')
    assert aoi_dir_name != None, "Environment variable AOI_DIR_NAME is None"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    idx = current_dir.find(aoi_dir_name) + len(aoi_dir_name)
    aoi_dir = current_dir[:idx]


    ### train yaml
    # Write config_file
    config_file = os.path.join(aoi_dir, 'YOLO_train/cfg/yolov4.yaml')
    config_dict = read_config_yaml(config_file)
    
    # set path
    config_dict['YOLO_darknet_path']    = os.path.join(aoi_dir, 'darknet')
    config_dict['YOLO_config_path'] = os.path.join(current_dir, 'cfg')
    config_dict['YOLO_weight_path'] = os.path.join(current_dir, 'weights')
    config_dict['YOLO_dataset_path']= os.path.join(current_dir, 'yolov4_dataset')
    config_dict['YOLO_inference_path']   = os.path.join(current_dir, 'result')

    config_dict['yolov4_model_file_path']  = os.path.join(config_dict['YOLO_config_path'], 'yolov4.conv.137')
    config_dict['test_GT_csv']  = os.path.join(config_dict['YOLO_config_path'] , 'test_GT.csv')
    config_dict['valid_GT_csv'] = os.path.join(config_dict['YOLO_config_path'] , 'valid_GT.csv')

    config_dict['yolo_valid_csv'] = 'yolo_valid.csv'
    
    config_dict['yolov4_best_model'] = config_dict['yolov4_best_model']


    config_dict['TRAIN_data_path']  = os.path.join(config_dict['YOLO_dataset_path'], 'train_data')
    config_dict['VAL_data_path']    = os.path.join(config_dict['YOLO_dataset_path'], 'valid_data')
    config_dict['TEST_data_path']   = os.path.join(config_dict['YOLO_dataset_path'], 'valid_data')


    config_dict['valid_eval_csv']   = os.path.join(config_dict['YOLO_inference_path'], 'valid_eval.csv')
    config_dict['test_eval_csv']   = os.path.join(config_dict['YOLO_inference_path'], 'test_eval.csv')



    # customized config from ENV 
    config_dict['ORI_dataset_path']     = check_env(os.path.join(aoi_dir, 'dataset'),'ORI_dataset_path')
    config_dict['project_name']         = check_env("yolov4_train", "project_name")
    config_dict['input_dataset_path']   = os.path.join(config_dict['ORI_dataset_path'], config_dict['project_name'])

    # set config 
    # train
    config_dict['set_ratio']    = check_env(0.6,'set_ratio')
    config_dict['batch']        = check_env(16,'batch')
    config_dict['subdivisions'] = check_env(4,'subdivisions')
    config_dict['width']        = check_env(512,'width')
    config_dict['height']       = check_env(512,'height')
    config_dict['channels']     = check_env(3,'channels')
    config_dict['max_batches']  = check_env(10000,'max_batches')
    config_dict['select_cfg']   = check_env('yolov4.cfg','select_cfg')
    config_dict['learning_rate'] = check_env(-1,'learning_rate')

    # inference
    config_dict['NMS_flag']             = check_env(1,'NMS_flag')
    config_dict['NMS_Iou_threshold']    = check_env(0.75,'NMS_Iou_threshold')
    config_dict['Edge_limit']           = check_env(10,'Edge_limit')
    config_dict['inference_Batch_size'] = check_env(1,'inference_Batch_size')
    config_dict['Score_threshold']      = check_env(0.001,'Score_threshold')
    config_dict['Iou_threshold']        = check_env(0.1,'Iou_threshold')

    # model performance
    config_dict['Precision']= check_env(0,'Precision')
    config_dict['Recall']   = check_env(0,'Recall')
    config_dict['F1_score'] = check_env(0,'F1_score')





    print(dict(config_dict))
    write_config_yaml(config_file, config_dict)
