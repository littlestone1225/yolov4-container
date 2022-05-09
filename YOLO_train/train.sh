#!/bin/bash
# Absolute path to this script file
SCRIPT_FILE=$(readlink -f "$0")

# Absolute directory this script is in
SCRIPT_DIR=$(dirname "$SCRIPT_FILE")

# Absolute path to the AOI_PCB directory
AOI_PCB_DIR=$(dirname "$SCRIPT_DIR")

# Get ensemble_config.yaml and config from ensemble_config.py
function Fun_ConvertConfigResult()
{
    config_result=$1
    print_result=$2
    config_result=`echo $config_result | sed 's/{//g' | sed 's/}//g' | sed 's/'\''//g'`
    IFS=",:" read -a config <<< $config_result

    for (( i=0; i<${#config[@]}; i+=2 ))
    do
        key=`echo ${config[$i]^^} | sed -e 's/\r//g'`
        value=`echo ${config[$i+1]} | sed -e 's/\r//g'`
        eval "$key=$value"
        if [ "$print_result" = "1" ]
        then
            printf "${YELLOW}%-20s = $value${NC}\n" $key
        fi
    done
}

# 1. generate yolov4 config yaml 
printf "python $AOI_PCB_DIR/YOLO_train/yolo_config_yaml.py${NC}\n"
config_result=`python $AOI_PCB_DIR/YOLO_train/yolo_config_yaml.py`
Fun_ConvertConfigResult "$config_result" 1
echo "$YOLO_DARKNET_PATH"/darknet detector calc_anchors "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".data -num_of_clusters 9 -width "$WIDTH" -height "$HEIGHT"

"$YOLO_DARKNET_PATH"/darknet detector calc_anchors "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".data -num_of_clusters 9 -width "$WIDTH" -height "$HEIGHT"



# 2. retrain preprocess
python3 yolov4_pre_process.py


# 3. calculate anchor
python3 set_yolo_anchor.py


# 4. yolov4 training
if [ -z "$var" ]
then
    YOLOV4_BEST_MODEL="$YOLOV4_MODEL_FILE_PATH"  
fi
echo "$YOLO_DARKNET_PATH"/darknet detector train "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".data "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".cfg "$YOLOV4_BEST_MODEL" -dont_show -clear 1 #-gpus 1 
"$YOLO_DARKNET_PATH"/darknet detector train "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".data "$YOLO_CONFIG_PATH"/"$PROJECT_NAME".cfg "$YOLOV4_BEST_MODEL" -dont_show -clear 1 #-gpus 1 

