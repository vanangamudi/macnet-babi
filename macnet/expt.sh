#!/bin/sh

for i in $(seq 5 20)
do
    echo $(date)
    time python main.py --hpconfig hpconfig$i.py  train
done
