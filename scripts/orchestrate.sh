#!/bin/bash
# for farming out modeling jobs the e-z way

# all credit to @pkuczynski https://gist.github.com/pkuczynski/8665367
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_\_]*' fs=$(echo @|tr @ '\034')
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
MEMUSAGE=2000

OPTION_ARRAY=("WALLTIME" "DIR" "SUBCOMMAND" "CONFIG" "INPUT" "OUTPUT" "QUEUE" "LOG" "NPROCS" "OPTIONS" "DRYRUN" "MEMUSAGE")

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
    -m|--mem-usage)
		MEMUSAGE="$2"
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

	eval $(parse_yaml $CONFIG "YAML_")

	for i in "${OPTION_ARRAY[@]}"; do
		a="YAML_bash_$i"
		if [[ ${!a:+1} ]]; then
			echo "Setting" $i "to" ${!a}
			declare $i="${!a}"
		fi
	done

fi

INPUT="$DIR"/$(basename "$INPUT")
OUTPUT="$DIR"/$(basename "$OUTPUT")
LOG="$DIR"/$(basename "$LOG")

# issue ze command

echo "bsub -q $QUEUE -W $WALLTIME -R \"rusage[mem=$MEMUSAGE]\" -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS"
if [ "$DRYRUN" = false ]; then
	bsub -q $QUEUE -W $WALLTIME -R "rusage[mem=$MEMUSAGE]" -o $LOG -N -n $NPROCS mpirun -n $NPROCS kinect_model $SUBCOMMAND $CONFIG $INPUT $OUTPUT $OPTIONS
fi
