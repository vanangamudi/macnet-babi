mkdir ~/projects/logs/run02
bash expt.sh reasoning_steps run02 2>&1 | tee ~/projects/logs/run02/reasoning_steps.log

mkdir ~/projects/logs/run03
bash expt.sh reasoning_steps run03 2>&1 | tee ~/projects/logs/run03/reasoning_steps.log

mkdir ~/projects/logs/run04
bash expt.sh reasoning_steps run04 2>&1 | tee ~/projects/logs/run04/reasoning_steps.log
