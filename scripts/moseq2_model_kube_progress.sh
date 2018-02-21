#!/bin/bash
# for farming out modeling jobs the e-z way

hash kubectl 2>/dev/null || { echo >&2 "kubectl not installed.  Aborting."; exit 1; }
hash tqdm 2>/dev/null || { echo >&2 "tqdm not installed.  Aborting."; exit 1; }

AUTHORIZE=false

if [ -z "${KINECT_GKE_PROGRESS_INTERVAL}" ]; then
	INTERVAL=1
else
	echo "Setting interval to ${KINECT_GKE_PROGRESS_INTERVAL}"
	INTERVAL=${KINECT_GKE_PROGRESS_INTERVAL}
fi

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -i|--interval)
    INTERVAL="$2"
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

# this is super-incomplete, use at your own risk
# note that wc -l  will return an extra line
loop () {

# just convert to a spinner with n/nn jobs completed

COMPLETED=0
while true; do

	COUNTER=0
	PRINTCOUNTER=0

	while IFS='' read -r line; do
		(( COUNTER+=1 ))
		if [[ "$COUNTER" -gt "$COMPLETED" ]]; then
			printf "%s\n" "$line"
		fi
	done < <(kubectl get pods -a | grep Completed)

	(( COMPLETED += COUNTER ))
	sleep ${INTERVAL}s

done


}

#USE_FILE=$(mktemp)
loop | tqdm --unit files --delim '\0' > /dev/null
#loop
#tail -f ${USE_FILE} > $(mktemp
