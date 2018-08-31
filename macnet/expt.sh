#!/bin/sh

echo $(date)
time python main.py --hpconfig hpconfig.py  train

for i in $(seq 1 20)
do
    echo $(date)
    time python main.py --hpconfig hpconfig$i.py  train && echo (date)
done

time python main.py --hpconfig hpconfig_10percent_dataset.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_20percent_dataset.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_30percent_dataset.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_40percent_dataset.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_50percent_dataset.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_75percent_dataset.py  train  && echo $(date)
								     && echo $(date)
time python main.py --hpconfig hpconfig_1_reasoning_steps.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_3_reasoning_steps.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_4_reasoning_steps.py  train  && echo $(date)
time python main.py --hpconfig hpconfig_5_reasoning_steps.py  train  && echo $(date)
								     && echo $(date)
time python main.py --hpconfig hpconfig_no_graph_reasoning.py train  && echo $(date)
time python main.py --hpconfig hpconfig_no_prev_mem.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_no_same_rnn.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_no_story_again.py     train  && echo $(date)
								     && echo $(date)
time python main.py --hpconfig hpconfig_story_len_10.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_story_len_20.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_story_len_30.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_story_len_40.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_story_len_50.py	      train  && echo $(date)
time python main.py --hpconfig hpconfig_story_len_60.py	      train  && echo $(date)

