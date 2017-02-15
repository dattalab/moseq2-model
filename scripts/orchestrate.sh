#!/bin/bash
# for farming out modeling jobs the e-z way

# all credit to @pkuczynski https://gist.github.com/pkuczynski/8665367
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

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

OPTION_ARRAY=("WALLTIME" "DIR" "SUBCOMMAND" "CONFIG" "INPUT" "OUTPUT" "QUEUE" "LOG" "NPROCS" "OPTIONS" "DRYRUN")

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

CONFIG="$DIR"/$(basename "$CONFIG")

# read yaml file if it exists

if [ -e $CONFIG ]; then

	YAML=parse_yaml($CONFIG)

	for i in "${OPTION_ARRAY[@]}"; do
		if [-n "$YAML_bash_$i"]; then
			$i=$YAML_bash_$i
		fi
	done

	# if [ -n "$YAML_bash_input" ]; then
	# 	$INPUT=$YAML_bash_input
	# fi
	#
	# if [ -n "$YAML_bash_output" ]; then
	# 	$OUTPUT=$YAML_bash_output
	# fi
	#
	# if [ -n "$YAML_bash_log" ]; then
	# 	$LOG=$YAML_bash_log
	# fi
	#
	# if [ -n "$YAML_bash_queue" ]; then
	# 	$QUEUE=$YAML_bash_queue
	# fi
	#
	# if [ -n "$YAML_bash_subcommand" ]; then
	# 	$SUBCOMMAND=$YAML_bash_subcommand
	# fi
	#
	# if [ -n "$YAML_bash_nprocs" ]; then
	# 	$NPROCS=$YAML_bash_nprocs
	# fi
	#
	# if [ -n "$YAML_bash_walltime" ]; then
	# 	$WALLTIME=$YAML_bash_walltime
	# fi
	#
	# if [ -n "$YAML_bash_options" ]; then
	# 	$OPTIONS=$YAML_bash_options
	# fi

fi

INPUT="$DIR"/$(basename "$INPUT")
OUTPUT="$DIR"/$(basename "$OUTPUT")
LOG="$DIR"/$(basename "$LOG")

# issue ze command

echo "bsub -q $QUEUE -W $WALLTIME -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS"
if [ "$DRYRUN" = false ]; then
	bsub -q $QUEUE -W $WALLTIME -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS
fi
