_kinect_model_completion() {
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _KINECT_MODEL_COMPLETE=complete $1 ) )
    return 0
}

complete -F _kinect_model_completion -o default kinect_model;
