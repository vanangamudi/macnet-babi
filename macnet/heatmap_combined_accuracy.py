import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
#plt.style.use('ggplot')
import pickle
import glob
import numpy as np

task_names = {
    1:'single-supporting-fact',
    2:'two-supporting-facts',
    3:'three-supporting-facts',
    4:'two-arg-relations',
    5:'three-arg-relations' ,
    6:'yes-no-questions',
    7:'counting',
    8:'lists-sets',
    9:'simple-negation',
    10:'indefinite-knowledge',
    11:'basic-coreference',
    12:'conjunction',
    13:'compound-coreference',
    14:'time-reasoning',
    15:'basic-deduction',
    16:'basic-induction',
    17:'positional-reasoning',
    18:'size-reasoning',
    19:'path-finding',
    20:'agents-motivations',
    21: 'main'
}

task_ids = {v:k for k,v in task_names.items()}

colors = {
    1: '#DC143C',
    2: '#8B0000',
    3: '#FF1493',
    4: '#DB7093',
    5: '#FF6347',
    6: '#FF4500',
    7: '#FFFF00',
    8: '#FFDAB9',
    9: '#EE82EE',
    10: '#FF00FF',
    11: '#663399',
    12: '#4B0082',
    13: '#ADFF2F',
    14: '#00FF00',
    15: '#2E8B57',
    16: '#808000',
    17: '#008080',
    18: '#00FFFF',
    19: '#00BFFF',
    20: '#2F4F4F',
    21: '#000000',
}

epoch_limit = 400
    
def read_pkls():
    
    accuracies = {}
    max_epoch_count = 0
    min_epoch_count = 10000000
    
    hpconfig = 'hpconfig'
    HPCONFIG = importlib.__import__(hpconfig)
    tasks = '-'.join(str(i) for i in HPCONFIG.CONFIG.tasks)
    root_dir= initialize_task(hpconfig + '.py')
    print('root_dir: {}'.format(root_dir))
    for filename in glob.glob('{}/results/metrics/*.accuracy.pkl'.format(root_dir)):
        try:
            task = os.path.basename(filename).split('.')[0]
            accuracies[task] = pickle.load(
                open(filename,
                     'rb')
            )

            if len(accuracies[task]) < min_epoch_count:
                min_epoch_count = len(accuracies[task])
                print('min_epoch_count: {}'.format(min_epoch_count))


            if len(accuracies[task]) > max_epoch_count:
                max_epoch_count = len(accuracies[task])
                print('max_epoch_count: {}'.format(max_epoch_count))
        except:
            print('{} not found'.format(filename))

    return accuracies, min_epoch_count, max_epoch_count



def plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    plot_title='Combined Accuracy',
                    plot_filepath='combined_accuracy_heatmap.png',
):
    right_most_boundary = min(epoch_limit, min_epoch_count)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, right_most_boundary//10), 5))

    # Remove the plot frame lines. They are unnecessary here.
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    fig.subplots_adjust(left=.1, bottom=.02)
    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.

    ax.set_xlim(0, right_most_boundary)
    #ax.set_ylim(0.98, 1.01)

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    #plt.xticks(range(1970, 2011, 10), fontsize=14)
    #plt.yticks(range(0, 91, 10), fontsize=14)
    #ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    #plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted.
    #plt.tick_params(axis='both', which='both', bottom='off', top='off',
    #                labelbottom='on', left='off', right='off', labelleft='on')

    labels = []
    matrix = []
    for i, (task_name, acc) in  enumerate(accuracies_):
        labels.append('{}({:0.2f}/{:0.2f})'.format(task_name, max(acc), acc[right_most_boundary-1]))
        matrix.append(acc[:right_most_boundary])

    plt.sca(ax)

    matrix = np.asarray(matrix)
    aspect_ratio = max(1, matrix.shape[1]/matrix.shape[0])
    print(matrix.shape, aspect_ratio)
    im = ax.imshow(np.asarray(matrix, dtype=np.float), clim=(0,1), cmap='autumn',
                   aspect=aspect_ratio)
    print(labels)
    plt.yticks(range(len(labels)),labels)

    fig.colorbar(im, orientation='vertical', aspect=aspect_ratio)

    # Make the title big enough so it spans the entire plot, but don't make it
    # so big that it requires two lines to show.

    # Note that if the title is descriptive enough, it is unnecessary to include
    # axis labels; they are self-evident, in this plot's case.
    fig.suptitle(plot_title, fontsize=18, ha='center')

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls()
    accuracies = sorted(accuracies.items(), key=lambda x: task_ids[x[0]])
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    'Combined Accuracy',
                    'combined_accuracy_heatmap.png')
