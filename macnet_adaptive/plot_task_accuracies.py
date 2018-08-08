import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import importlib
from anikattu.utilz import initialize_task
plt.style.use('ggplot')
import pickle

task_names = {
    1:' single-supporting-act',
    2:' two-supporting-facts',
    3:' three-supporting-facts',
    4:' two-arg-relations',
    5:' three-arg-relations' ,
    6:' yes-no-questions',
    7:' counting',
    8:' list-sets',
    9:' simple-negation',
    10:' indefinite-knowledge',
    11:' basic-coreference',
    12:' conjuction',
    13:' compound-coreference',
    14:' time-reasoning',
    15:' basic-deduction',
    16:' basic-induction',
    17:' positional-reasoning',
    18:' size-reasoning',
    19:' path-finding',
    20:' agent-motivations',
}

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

root_dirs = {}
accuracies = {}
max_epoch_count = 0
epoch_limit = 200
for hpconfig in hpconfigs:
    HPCONFIG = importlib.__import__(hpconfig)
    tasks = '-'.join(str(i) for i in HPCONFIG.CONFIG.tasks)
    root_dirs[tasks] = initialize_task(hpconfig + '.py')
    accuracies[tasks] = pickle.load(
        open('{}/results/metrics/main.accuracy.pkl'.format(root_dirs[tasks]),
             'rb')
    )

    if len(accuracies[tasks]) > 50:
        accuracies[tasks][50] = accuracies[tasks][49]
        accuracies[tasks][100] = accuracies[tasks][99]
        accuracies[tasks][150] = accuracies[tasks][149]
        
        accuracies[tasks][51] = accuracies[tasks][50]
        accuracies[tasks][101] = accuracies[tasks][100]
        accuracies[tasks][151] = accuracies[tasks][150]
    
    if len(accuracies[tasks]) > max_epoch_count:
        max_epoch_count = len(accuracies[tasks])
        print('max_epoch_count: {}'.format(max_epoch_count))

    
# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# You typically want your plot to be ~1.33x wider than tall. This plot
# is a rare exception because of the number of lines being plotted on it.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 14))

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
ax.set_xlim(0, epoch_limit)
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

for i, (task_name, acc) in  enumerate(accuracies.items()):
    line = plt.plot(acc[:epoch_limit-1],
                    lw=2.5,
                    color=colors[i+1])

    # Add a text label to the right end of every line. Most of the code below
    # is adding specific offsets y position because some labels overlapped.
    acc_ = acc[:epoch_limit]
    y_pos = acc_[-1] #- 0.5
    """
    if column in y_offsets:
        y_pos += y_offsets[column]
    """
    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    try:
        name = task_names[int(task_name)]
    except:
        name = 'ALL'
    plt.text(epoch_limit + 0.5 , y_pos, name , fontsize=14, color=colors[i+1])

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
fig.suptitle('Percentage of Bachelor\'s degrees conferred to women in '
             'the U.S.A. by major (1970-2011)\n', fontsize=18, ha='center')

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# plt.savefig('percent-bachelors-degrees-women-usa.png', bbox_inches='tight')
plt.show()
