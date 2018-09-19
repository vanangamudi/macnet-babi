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
    'hpconfig_story_len_10',
    'hpconfig_story_len_20',
    'hpconfig_story_len_30',
    'hpconfig_story_len_40',
    'hpconfig_story_len_50',
    'hpconfig_story_len_60',
    'hpconfig' ,                 # all
]


if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls(hpconfigs, 'hpconfig_story_len_(\d+)')
    pprint(accuracies)
    labels = {k:'story_len = {}'.format(k) for k in accuracies.keys()}
    labels['main'] = 'main'
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies.items(), task_ids,
                    'Story length',
                    'story_len_training_accuracy.png',
                    labels = labels,
                    y_offsets = {},
)
