export KINECT_GKE_MODEL_IMAGE=gcr.io/kinect-modeling/kinect-modeling
export KINECT_GKE_MODEL_BUCKET=kinect-modeling-$USER
export KINECT_GKE_MODEL_NCPUS=4
export KINECT_GKE_MOUNT_POINT=/mnt/modeling_bucket
export KINECT_GKE_LOG_PATH=~/Desktop/kubejobs
export KINECT_GKE_CLUSTER_NAME=modeling-cluster-$USER
export KINECT_GKE_MACHINE_TYPE=n1-highcpu-4
export KINECT_GKE_NUM_NODES=6
