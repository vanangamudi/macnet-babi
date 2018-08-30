#!/bin/sh

for i in $(seq 1 20)
do
    echo $(date)
    time python main.py --hpconfig hpconfig$i.py  train
done
