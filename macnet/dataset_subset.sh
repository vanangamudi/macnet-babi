mkdir ~/projects/logs/run02
bash expt.sh dataset_subset run02 2>&1 | tee ~/projects/logs/run02/dataset_subset.log

mkdir ~/projects/logs/run03
bash expt.sh dataset_subset run03 2>&1 | tee ~/projects/logs/run03/dataset_subset.log

mkdir ~/projects/logs/run04
bash expt.sh dataset_subset run04 2>&1 | tee ~/projects/logs/run04/dataset_subset.log
