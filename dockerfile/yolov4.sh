#!/bin/bash

# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'


USER=$USER
UIDCOSTUM=$UID
GIDCOSTUM=$(id -g "$USER")
echo "USER  = "$USER
echo "UIDCOSTUM  = "$UIDCOSTUM
echo "GIDCOSTUM  = "$GIDCOSTUM

# Absolute path to this script.
# e.g. /home/ubuntu/AOI_PCB_Inference/dockerfile/dockerfile_inference.sh
SCRIPT_PATH=$(readlink -f "$0")

# Absolute path this script is in.
# e.g. /home/ubuntu/AOI_PCB_Inference/dockerfile
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Absolute path to the AOI path
# e.g. /home/ubuntu/AOI_PCB_Inference
HOST_AOI_DIR=$(dirname "$SCRIPT_DIR")
echo "HOST_AOI_DIR  = "$HOST_AOI_DIR

# AOI directory name
IFS='/' read -a array <<< $HOST_AOI_DIR

AOI_DIR_NAME="${array[-1]}"
echo "AOI_DIR_NAME   = "$AOI_DIR_NAME


VERSION=$2
if [ "$2" == "" ]
then
    VERSION="v1.0"
else
    VERSION=$2
fi
echo "VERSION        = "$VERSION

IMAGE_NAME="littlestone1225/yolov4:$VERSION"
CONTAINER_NAME="littlestone1225_yolov4_$VERSION"
INFERENCE_CONTAINER_NAME="littlestone1225_yolov4_inference"
echo "IMAGE_NAME               = "$IMAGE_NAME
echo "CONTAINER_NAME           = "$CONTAINER_NAME
echo "INFERENCE_CONTAINER_NAME = "$INFERENCE_CONTAINER_NAME

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}${cmd}${NC}\n"
        eval $cmd
    done
}



# process command
if [ "$1" == "build" ]
then
    export GID=$(id -g)

    lCmdList=(
                "docker build \
                    --build-arg USER=$USER \
                    --build-arg UID=$UIDCOSTUM \
                    --build-arg GID=$GIDCOSTUM \
                    --build-arg AOI_DIR_NAME=$AOI_DIR_NAME \
                    -f yolov4.dockerfile \
                    -t $IMAGE_NAME ."
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "run" ]
then
    HOST_API_PORT="80"

    lCmdList=(
                "docker run --gpus all -it \
                    --privileged \
                    --restart unless-stopped \
                    --ipc=host \
                    --name $CONTAINER_NAME \
                    -v $HOST_AOI_DIR:/home/$USER/$AOI_DIR_NAME \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /etc/localtime:/etc/localtime:ro \
                    --env-file /home/$USER/$AOI_DIR_NAME/dockerfile/env.list \
                    --mount type=bind,source=$SCRIPT_DIR/.bashrc,target=/home/$USER/.bashrc \
                    $IMAGE_NAME /home/$USER/$AOI_DIR_NAME/dockerfile/run_container.sh" 
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "inference" ]
then
    HOST_API_PORT="80"

    lCmdList=(
                "docker run --gpus all -it \
                    --privileged \
                    --ipc=host \
                    --name $INFERENCE_CONTAINER_NAME \
                    -v $HOST_AOI_DIR:/home/$USER/$AOI_DIR_NAME \
                    -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /etc/localtime:/etc/localtime:ro \
                    --mount type=bind,source=$SCRIPT_DIR/.bashrc,target=/home/$USER/.bashrc \
                    --env-file /home/$USER/$AOI_DIR_NAME/dockerfile/env.list \
                    $IMAGE_NAME /home/$USER/$AOI_DIR_NAME/dockerfile/run_inference.sh" 
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "exec" ]
then
    lCmdList=(
                "docker exec -it $CONTAINER_NAME /bin/bash"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "start" ]
then
    lCmdList=(
                "docker start -ia $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "attach" ]
then
    lCmdList=(
                "docker attach $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "stop" ]
then
    lCmdList=(
                "docker stop $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rm" ]
then
    lCmdList=(
                "docker rm $CONTAINER_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "rmi" ]
then
    lCmdList=(
                "docker rmi $IMAGE_NAME"
             )
    Fun_EvalCmd "${lCmdList[*]}"

elif [ "$1" = "clean" ]
then
    lCmdList=(
                "sudo rm ../YOLO_train/cfg/train.txt"
                "sudo rm ../YOLO_train/cfg/valid.txt"
                "sudo rm ../YOLO_train/cfg/valid_GT.csv"
                "sudo rm ../YOLO_train/cfg/yolov4.yaml.lock"
                "sudo rm ../YOLO_train/anchors.txt"
                "sudo rm ../YOLO_train/chart*"
                "sudo rm ../YOLO_train/counters_per_class.txt"
                "sudo rm ../YOLO_train/bad.list"
                "sudo rm -r ../YOLO_train/yolov4_dataset/"
                "sudo rm -r ../YOLO_train/result/"
                "sudo rm -r ../YOLO_train/weights/"
                "sudo rm -r ../YOLO_train/__pycache__/"
                #"sudo rm -r ../darknet/"
             )
    Fun_EvalCmd "${lCmdList[*]}"

fi
