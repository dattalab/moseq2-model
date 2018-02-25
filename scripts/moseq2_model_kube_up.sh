#!/bin/bash
# for farming out modeling jobs the e-z way

hash gcloud 2>/dev/null || { echo >&2 "gcloud not installed.  Aborting."; exit 1; }

#######################################
# Run extraction
# Environment variables (optional, will override defaults for options):
#   MOSEQ2_GKE_CLUSTER_NAME=cluster-name
#   MOSEQ2_GKE_MACHINE_TYPE=machine-type
#   MOSEQ2_GKE_NUM_NODES=num-nodes
#   MOSEQ2_GKE_SCOPES=scopes
#   MOSEQ2_GKE_AUTOSCALE=autoscale
#   MOSEQ2_GKE_PREEMPTIBLE=preemptbile
#
# Arguments:
#   -c|--cluster-name (string): name of cluster"
#   -m|--machine-type (string): gce machine type default n1-standard-2
#   -n|--num-numnodes (int): number of nodes to start (default: 3)
#   -e|--environment (string): cluster environment (o2 or orchestra)
#   -s|--scope (string): gce scope (default: storage-full)
#   -a|--authorize (flag): run gce authentication
#   -p|--preemptible (flag): use preemptible nodes
#   -d|--dry-run (flag): display command to be issued without actually running
#
# Returns:
#   None
#
# Example (to show the command you would run):
# $ moseq2_model_kube_up -n 12 --cluster-name kinect-modeling
#
#######################################

if [ -z "${MOSEQ2_GKE_CLUSTER_NAME}" ]; then
	CLUSTERNAME="test-cluster"
else
	echo "Setting cluster name to ${MOSEQ2_GKE_CLUSTER_NAME}"
	CLUSTERNAME="${MOSEQ2_GKE_CLUSTER_NAME}"
fi

if [ -z "${MOSEQ2_GKE_MACHINE_TYPE}" ]; then
	MACHINETYPE="n1-standard-2"
else
	echo "Setting machine type to ${MOSEQ2_GKE_MACHINE_TYPE}"
	MACHINETYPE=${MOSEQ2_GKE_MACHINE_TYPE}
fi

if [ -z "${MOSEQ2_GKE_NUM_NODES}" ]; then
	NUMNODES="3"
else
	echo "Setting number of nodes to ${MOSEQ2_GKE_NUM_NODES}"
	NUMNODES=${MOSEQ2_GKE_NUM_NODES}
fi

if [ -z "${MOSEQ2_GKE_SCOPES}" ]; then
	SCOPES="storage-full"
else
	echo "Setting scopes to ${MOSEQ2_GKE_SCOPES}"
	SCOPES=${MOSEQ2_GKE_SCOPES}
fi

AUTHORIZE=false

if [ -z "${MOSEQ2_GKE_PREEMPTIBLE}" ]; then
	PREEMPTIBLE=false
else
	echo "Setting preemptible to ${MOSEQ2_GKE_PREEMPTIBLE}"
	PREEMPTIBLE=${MOSEQ2_GKE_PREEMPTIBLE}
fi

if [ -z "${MOSEQ2_GKE_AUTOSCALE}" ]; then
	AUTOSCALE=false
else
	echo "Setting autoscale to ${MOSEQ2_GKE_AUTOSCALE}"
	AUTOSCALE=${MOSEQ2_GKE_AUTOSCALE}
fi

DRYRUN=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: $0 -i [options]";
    echo "";
    echo "Arguments:";
    echo "  -c|--cluster-name (string): name of cluster"
    echo "  -m|--machine-type (string): gce machine type default n1-standard-2"
    echo "  -n|--num-numnodes (int): number of nodes to start (default: 3)"
    echo "  -e|--environment (string): cluster environment (o2 or orchestra)"
    echo "  -s|--scope (string): gce scope (default: storage-full)"
    echo "  -a|--authorize (flag): run gce authentication"
    echo "  -p|--preemptible (flag): use preemptible nodes"
    echo "  -d|--dry-run (flag): display command to be issued without actually running"
    echo "  --autoscale (flag): use gke autoscaler"
    echo "";
    exit 1;
    ;;
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
		;;
		-p|--preemptible)
		PREEMPTIBLE=true
		;;
		-d|--dry-run)
		DRYRUN=true
		;;
		--auto-scale)
		AUTOSCALE=true
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

COMMAND="gcloud container clusters create ${CLUSTERNAME} --scopes ${SCOPES} --machine-type ${MACHINETYPE} --num-nodes ${NUMNODES}"

if [ "${PREEMPTIBLE}" = true ]; then
	COMMAND+=" --preemptible"
fi

if [ "${AUTOSCALE}" = true ]; then
	COMMAND+=" --enable-autoscaling --min-nodes 1 --max-nodes ${NUMNODES}"
fi

CREDENTIALS="gcloud container clusters get-credentials ${CLUSTERNAME}"

echo $COMMAND
echo $CREDENTIALS

if [ "${DRYRUN}" = false ]; then

	eval $COMMAND
	eval $CREDENTIALS
	# mos def gotta kill kubectl proxy if it exists

	PROXY_PID=$(pgrep -f "kubectl proxy")
	if [ ! -z $PROXY_PID ]; then
		echo "Killing proxy PID $PROXY_PID"
		kill $PROXY_PID
	fi

	kubectl proxy >/dev/null 2>&1 &

	# copy the proxy pid

	sleep 10
	echo "Opening browser window for monitoring cluster"
	open http://localhost:8001/ui

fi
