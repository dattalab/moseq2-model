#!/bin/bash
# for farming out modeling jobs the e-z way

hash gcloud 2>/dev/null || { echo >&2 "gcloud not installed.  Aborting."; exit 1; }

if [ -z "${KINECT_CLUSTER_NAME}" ]; then
	CLUSTERNAME="test-cluster"
else
	echo "Setting cluster name to ${KINECT_CLUSTER_NAME}"
	CLUSTERNAME="${KINECT_CLUSTER_NAME}"
fi

if [ -z "${KINECT_MACHINE_TYPE}" ]; then
	MACHINETYPE="n1-standard-2"
else
	echo "Setting machine type to ${KINECT_MACHINE_TYPE}"
	MACHINETYPE=${KINECT_MACHINE_TYPE}
fi

if [ -z "${KINECT_NUM_NODES}" ]; then
	NUMNODES="3"
else
	echo "Setting number of nodes to ${KINECT_NUM_NODES}"
	NUMNODES=${KINECT_NUM_NODES}
fi

if [ -z "${KINECT_SCOPES}" ]; then
	SCOPES="storage-full"
else
	echo "Setting scopes to ${KINECT_SCOPES}"
	SCOPES=${KINECT_SCOPES}
fi

AUTHORIZE=false

if [ -z "${KINECT_PREEMPTIBLE}" ]; then
	PREEMPTIBLE=false
else
	echo "Setting preemptible to ${KINECT_PREEMPTIBLE}"
	PREEMPTIBLE=${KINECT_PREEMPTIBLE}
fi

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--cluster-name)
    CLUSTERNAME="$2"
    shift # past argument
    ;;
    -m|--machine-type)
    MACHINETYPE="$2"
    shift # past argument
    ;;
    -n|--num-nodes)
    NUMNODES="$2"
    shift # past argument
    ;;
		-s|--scope)
		SCOPE="$2"
		shift
		;;
		-a|--authorize)
		AUTHORIZE=true
		shift
		;;
		-p|--preemptible)
		PREEMPTIBLE=true
		;;
		*)
            # unknown option
    ;;
esac
shift
done

if [ "${AUTHORIZE}" = true ]; then
	gcloud auth application-default login
fi

# can make preemptible

COMMAND="gcloud container clusters create ${CLUSTERNAME} --scopes ${SCOPES} --machine-type ${MACHINETYPE} --num-nodes ${NUMNODES}"

if [ "${PREEMPTIBLE}" = true ]; then
	COMMAND+=" --preemptible"
fi

CREDENTIALS="gcloud container clusters get-credentials ${CLUSTERNAME}"

echo $COMMAND
echo $CREDENTIALS

#gcloud container clusters create test-cluster --scopes storage-full --machine-type n1-highcpu-8 --num-nodes 3
#gcloud container clusters get-credentials test-cluster

# tear down with gcloud container clusters delete test-cluster

#gcloud container clusters delete test-cluster
