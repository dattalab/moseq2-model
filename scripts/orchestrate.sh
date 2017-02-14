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

while [[ $# -gt 1 ]]
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
    ;;
		-i|--input)
		INPUT="$2"
		;;
		-q|--queue)
		QUEUE="$2"
		;;
		-l|--log)
		LOG="$2"
		;;
		-n|--nprocs)
		NPROCS="$2"
		;;
		-c|--config)
		CONFIG="$2"
		;;
		--options)
		OPTIONS="$2"
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
#bsub -q $QUEUE -w $WALLTIME -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS
