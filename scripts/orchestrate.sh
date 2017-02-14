#!/bin/bash
# for farming out modeling jobs the e-z way

WALLTIME="24:00"
DIR=$(pwd)
SUBCOMMAND="cv_parameter_scan"
CONFIG="config.yaml"
INPUT="input.mat"
OUTPUT="output.mat"
QUEUE="mpi"
LOG="job.out"
NPROCS="20"
OPTIONS=""
DRYRUN=false
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -w|--wall-time)
    WALLTIME="$2"
    shift # past argument
    ;;
    -d|--dir)
    DIR="$2"
    shift # past argument
    ;;
    -s|--subcommand)
    SUBCOMMAND="$2"
    shift # past argument
    ;;
    -o|--output)
    OUTPUT="$2"
		shift
    ;;
		-i|--input)
		INPUT="$2"
		shift
		;;
		-q|--queue)
		QUEUE="$2"
		shift
		;;
		-l|--log)
		LOG="$2"
		shift
		;;
		-n|--nprocs)
		NPROCS="$2"
		shift
		;;
		-c|--config)
		CONFIG="$2"
		shift
		;;
		--options)
		OPTIONS="$2"
		shift
		;;
		--dry-run)
		DRYRUN=true
		;;
    *)
            # unknown option
    ;;
esac
shift
done

# put it all together

INPUT="$DIR"/$(basename "$INPUT")
OUTPUT="$DIR"/$(basename "$OUTPUT")
LOG="$DIR"/$(basename "$LOG")
CONFIG="$DIR"/$(basename "$CONFIG")


# issue ze command

echo "bsub -q $QUEUE -w $WALLTIME -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS"
if [ "$DRYRUN" = false ]; then
	bsub -q $QUEUE -w $WALLTIME -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS
fi
