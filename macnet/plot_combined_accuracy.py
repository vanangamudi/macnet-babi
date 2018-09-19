import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
plt.style.use('ggplot')
import pickle
import glob

import numpy as np
from heatmap_combined_accuracy import task_names, task_ids
from heatmap_combined_accuracy import read_pkls

colors = {
    1: '#DC143C',
    2: '#8B0000',
    3: '#FF1493',
    4: '#DB7093',
    5: '#FF6347',
    6: '#FF3000',
    7: '#663399',
    8: '#FFDAB9',
    9: '#EE82EE',
    10: '#FF00FF',
    11: '#FFFF00',
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

epoch_limit = 200

def calc_moving_avg(p, N = 5):
    return np.convolve(p , np.ones((N,))/N, mode='same')[:-1]

def plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    plot_title='Combined Accuracy',
                    plot_filepath='combined_accuracy_heatmap.png',
                    labels = {},
                    y_offsets = {},
                    ylabel = 'Accuracy',
                    xlabel = 'Epoch',
                    ylim = (0, 1),
                    moving_avg = 0,
):
    # You typically want your plot to be ~1.33x wider than tall. This plot
    # is a rare exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Remove the plot frame lines. They are unnecessary here.
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    right_most_boundary = min(epoch_limit, max_epoch_count) + 1
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
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just
    # plotted.
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='on', left='off', right='off', labelleft='on')

    for i, (task_name, acc) in enumerate(accuracies):
        p = acc[: right_most_boundary]
        if moving_avg:
            p = calc_moving_avg(p, moving_avg)
            
        line = plt.plot(p,
                        lw=2.5,
                        color=colors[i+1])

        # Add a text label to the right end of every line. Most of the code below
        # is adding specific offsets y position because some labels overlapped.
        acc_ = acc[:right_most_boundary]
        y_pos = acc_[-1] #- 0.5


        # Again, make sure that all labels are large enough to be easily read
        # by the viewer.
        task_name = os.path.basename(task_name)

        if task_name in y_offsets:
            y_pos += y_offsets[task_name]

        if task_name in labels:
            task_name = labels[task_name]

        plt.text(right_most_boundary + 0.5 , y_pos,
                 '{}({:0.3f})'.format(task_name, acc_[-1]),
                 fontsize=14, color=colors[i+1])

    # Make the title big enough so it spans the entire plot, but don't make it
    # so big that it requires two lines to show.

    # Note that if the title is descriptive enough, it is unnecessary to include
    # axis labels; they are self-evident, in this plot's case.
    fig.suptitle(plot_title, fontsize=18, ha='center')

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    accuracies, min_epoch_count, max_epoch_count = read_pkls()
    accuracies = sorted(accuracies.items(), key=lambda x: task_ids[x[0]])
    plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies, task_ids,
                    'Combined Training Accuracy',
                    'combined_training_accuracy.png',
                    moving_avg = 7)
