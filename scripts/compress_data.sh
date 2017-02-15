#!/usr/bin/env sh

EXT="dat"
MTIME="+30"
LEVEL="6"
DRYRUN=false
KEEPFILE=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--ext)
    EXT="$2"
    shift # past argument
    ;;
		-m|--mtime)
    MTIME="$2"
    shift # past argument
    ;;
		-l|--level)
    LEVEL="$2"
    shift # past argument
    ;;
    -d|--dry-run)
    DRYRUN=true
    shift # past argument
    ;;
    -k|--keep-file)
    KEEPFILE=true
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift
done

FILELIST=(`find . -type f -name "*.$EXT" -mtime $MTIME`)

for i in "${FILELIST[@]}"; do

  echo "Will compress $i"
  BASEFILE=($(basename $i | cut -d. -f1))
  DIRNAME=($(dirname $i))
  echo "Saving md5 checksum to ${DIRNAME}/${BASEFILE}.md5"

  if [ "$DRYRUN" = false ]; then
    md5sum $i > ${DIRNAME}/${BASEFILE}.md5
  fi

  echo "Compressing to ${DIRNAME}/${BASEFILE}.gz"

  if [ "$DRYRUN" = false ]; then
    gzip < $i > ${DIRNAME}/${BASEFILE}.gz
  fi

  echo "Getting md5 from gzip archive"

  if [ "$DRYRUN" = false ]; then
    MD5ZIP=($(gzip -d -c ${DIRNAME}/${BASEFILE}.gz | md5sum))
    MD5=($(cat ${DIRNAME}/${BASEFILE}.md5))
    echo "Original MD5: ${MD5}"
    echo "gzip MD5: ${MD5ZIP}"
    if [ "$MD5ZIP" == "$MD5" ]; then
      echo "MD5 checks out"
      if [ "$KEEPFILE" = false ]; then
        echo "Deleting $i"
        rm $i
      fi
    fi
  fi

done
