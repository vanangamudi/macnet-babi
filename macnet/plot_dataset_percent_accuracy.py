import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
plt.style.use('ggplot')
import pickle
from pprint import pprint

from plot_combined_accuracy import task_names, task_ids, plot_accuracies
from plot_component_variations_accuracy import read_pkls

import re
root_dirs = {}
accuracies = {}
max_epoch_count = 0
min_epoch_count = 100000
epoch_limit = 200

hpconfigs = [
    'hpconfig_10percent_dataset',
    'hpconfig_20percent_dataset',
    'hpconfig_30percent_dataset',
    'hpconfig_40percent_dataset',
    'hpconfig_50percent_dataset',
    'hpconfig_75percent_dataset',
    'hpconfig'
]

if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls(hpconfigs,'hpconfig_(\d+)percent_dataset')
    pprint(accuracies)
    labels = {k:'size = {:0.2f}'.format(float(k)/100) for k in accuracies.keys() if k != 'main'}
    labels['main'] = 'main'
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies.items(), task_ids,
                    'Dataset Size',
                    'dataset_training_accuracy.png',
                    labels = labels,
                    y_offsets = {'10' : 0.0,
                                 '20' : 0.0,
                                 '30' : -0.005,
                                 '40' : 0.005,
                                 '50' : 0.005,
                                 '75' : -0.007,
                                 'main' : 0.007},
                    ylim = (0.6, 1)
    )
