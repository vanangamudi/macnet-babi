import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
plt.style.use('ggplot')
import pickle
from pprint import pprint

from plot_combined_accuracy import task_names, task_ids, plot_accuracies
from heatmap_task_accuracies import read_pkls
import re
root_dirs = {}
accuracies = {}
max_epoch_count = 0
min_epoch_count = 100000
epoch_limit = 200

hpconfigs = [
    'hpconfig_no_graph_reasoning',
    'hpconfig_no_prev_mem',
    'hpconfig_no_same_rnn',
    'hpconfig_no_story_again',
    'hpconfig'
]


def read_pkls(hpconfigs=hpconfigs, name_pattern='hpconfig_(no_\w+)'):
    root_dirs = {}
    accuracies = {}
    max_epoch_count = 0
    min_epoch_count = 10000000

    for hpconfig in hpconfigs:
        try:
            tasks = re.match(name_pattern, hpconfig)
            if tasks:
                tasks = tasks.group(1)
            else:
                tasks = 'main'
            root_dirs[tasks] = initialize_task(hpconfig + '.py')
            accuracies[tasks] = pickle.load(
                open('{}/results/metrics/main.accuracy.pkl'.format(root_dirs[tasks]),
                     'rb')
            )

            if len(accuracies[tasks]) < min_epoch_count:
                min_epoch_count = len(accuracies[tasks])
                print('min_epoch_count: {}'.format(min_epoch_count))

            if len(accuracies[tasks]) > max_epoch_count:
                max_epoch_count = len(accuracies[tasks])
                print('max_epoch_count: {}'.format(max_epoch_count))
        except:
            print('{} not found'.format(tasks))
            
    return accuracies, min_epoch_count, max_epoch_count


if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls()
    pprint(accuracies)
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies.items(), task_ids,
                      'Ablation study',
                    'components_removed_training_accuracy.png',
                    y_offsets = {'main': 0.0015, 'no_story_again': -0.0015, 'no_same_rnn' : -0.001},
                    ylim = (0.85, 1),
                    moving_avg = 4,
    )
