#!/bin/bash
# for farming out modeling jobs the e-z way

hash gcloud 2>/dev/null || { echo >&2 "gcloud not installed.  Aborting."; exit 1; }

if [ -z "${KINECT_GKE_CLUSTER_NAME}" ]; then
	CLUSTERNAME="test-cluster"
else
	echo "Setting cluster name to ${KINECT_GKE_CLUSTER_NAME}"
	CLUSTERNAME="${KINECT_GKE_CLUSTER_NAME}"
fi

AUTHORIZE=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--cluster-name)
    CLUSTERNAME="$2"
    shift # past argument
		;;
		-a|--authorize)
		AUTHORIZE=true
		shift
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

# tear down the cluster

COMMAND="gcloud container clusters delete ${CLUSTERNAME}"

echo $COMMAND
eval $COMMAND

PROXY_PID=$(pgrep -f "kubectl proxy")
if [ ! -z $PROXY_PID ]; then
	echo "Killing proxy PID $PROXY_PID"
	kill $PROXY_PID
fi
