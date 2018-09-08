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
    'hpconfig_1_reasoning_steps',
    'hpconfig_3_reasoning_steps',
    'hpconfig_4_reasoning_steps',
    'hpconfig_5_reasoning_steps',
]

if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls(hpconfigs, 'hpconfig_(\d+)_reasoning_steps')
    pprint(accuracies)
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies.items(), task_ids,
                    'Story Len X Accuracy',
                    'story_len_training_accuracy.png')
