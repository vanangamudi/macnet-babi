import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
#plt.style.use('ggplot')
import pickle
import numpy as np

from heatmap_combined_accuracy import task_names, task_ids, plot_accuracies

hpconfigs = [ 
    'hpconfig' ,    # all
    'hpconfig1',    # Single supporting act
    'hpconfig2',    # two supporting facts
    'hpconfig3',    # three supporting facts
    'hpconfig4',    # two arg relations
    'hpconfig5',    # three arg relations 
    'hpconfig6',    # yes-no-questions
    'hpconfig7',    # counting
    'hpconfig8',    # list-sets
    'hpconfig9',    # simple-negation
    'hpconfig10',   # indefinite-knowledge
    'hpconfig11',   # basic-coreference
    'hpconfig12',   # conjuction
    'hpconfig13',   # compound-coreference
    'hpconfig14',   # time-reasoning
    'hpconfig15',   # basic-deduction
    'hpconfig16',   # basic-induction
    'hpconfig17',   # positional-reasoning
    'hpconfig18',   # size-reasoning
    'hpconfig19',   # path-finding
    'hpconfig20',   # agent-motivations
]

epoch_limit = 500

def read_pkls(hpconfigs=hpconfigs):
    root_dirs = {}
    accuracies = {}
    max_epoch_count = 0
    min_epoch_count = 10000000

    for hpconfig in hpconfigs:
        try:
            HPCONFIG = importlib.__import__(hpconfig)
            if len(HPCONFIG.CONFIG.tasks) > 1:
                tasks = 'main'
            else:
                tasks = task_names[HPCONFIG.CONFIG.tasks[0]]

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
    
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    'Accuracies (individually trained)',
                    'individual_training_accuracy.png')
