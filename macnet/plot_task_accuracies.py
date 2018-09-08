import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
plt.style.use('ggplot')
import pickle

from plot_combined_accuracy import task_names, task_ids, plot_accuracies
from heatmap_task_accuracies import read_pkls

root_dirs = {}
accuracies = {}
max_epoch_count = 0
min_epoch_count = 100000
epoch_limit = 200


if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls()    
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    'Individual Training Accuracy',
                    'individual_training_accuracy.png')
