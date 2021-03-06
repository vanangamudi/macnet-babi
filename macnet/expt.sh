#!/bin/sh

echo "$1"
echo "$2"
echo "$@"


function normal {
    echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig.py  train

    for i in $(seq 1 20)
    do
	echo $(date)
	time python main.py  --prefix-dir $1 --hpconfig hpconfig$i.py  train && echo $(date)
    done
}

function dataset_subset {
    export CUDA_VISIBLE_DEVICES=0
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_10percent_dataset.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_20percent_dataset.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_30percent_dataset.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_40percent_dataset.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_50percent_dataset.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_75percent_dataset.py  train  && echo $(date)

}

function reasoning_steps {
    export CUDA_VISIBLE_DEVICES=1
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_1_reasoning_steps.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_3_reasoning_steps.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_4_reasoning_steps.py  train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_5_reasoning_steps.py  train  && echo $(date)
}

function component_variations {
    export CUDA_VISIBLE_DEVICES=2
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_no_graph_reasoning.py train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_no_prev_mem.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_no_same_rnn.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_no_story_again.py     train  && echo $(date)
}

function story_len {
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_10.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_20.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_30.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_40.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_50.py	      train  && echo $(date)
    time python main.py  --prefix-dir $1 --hpconfig hpconfig_story_len_60.py	      train  && echo $(date)

}

#https://stackoverflow.com/questions/8818119/how-can-i-run-a-function-from-a-script-in-command-line
echo "COMMAND: $@"
"$@"
