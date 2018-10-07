mkdir ~/projects/logs/run02
bash expt.sh component_variations run02 2>&1 | tee ~/projects/logs/run02/component_variations.log

mkdir ~/projects/logs/run03
bash expt.sh component_variations run03 2>&1 | tee ~/projects/logs/run03/component_variations.log

mkdir ~/projects/logs/run04
bash expt.sh component_variations run04 2>&1 | tee ~/projects/logs/run04/component_variations.log
