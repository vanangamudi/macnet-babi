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

accuracies = {}
max_epoch_count = 0
min_epoch_count = 1000000

epoch_limit = 400

hpconfig = 'hpconfig'
HPCONFIG = importlib.__import__(hpconfig)
tasks = '-'.join(str(i) for i in HPCONFIG.CONFIG.tasks)
root_dir= initialize_task(hpconfig + '.py')

for filename in glob.glob('{}/results/metrics/*.accuracy.pkl'.format(root_dir)):
    task = os.path.basename(filename).split('.')[0]
    accuracies[task] = pickle.load(
        open(filename,
             'rb')
    )

    if len(accuracies[task]) > 50:
        accuracies[task][50] = accuracies[task][49]
        accuracies[task][51] = accuracies[task][50]
    if len(accuracies[task]) > 100:
        accuracies[task][100] = accuracies[task][99]
        accuracies[task][101] = accuracies[task][100]
    if len(accuracies[task]) > 150:
        accuracies[task][150] = accuracies[task][149]
        accuracies[task][151] = accuracies[task][150]

    if len(accuracies[task]) < min_epoch_count:
        min_epoch_count = len(accuracies[task])
        print('min_epoch_count: {}'.format(min_epoch_count))
        
    if len(accuracies[task]) > max_epoch_count:
        max_epoch_count = len(accuracies[task])
        print('max_epoch_count: {}'.format(max_epoch_count))

        
    
right_most_boundary = min(epoch_limit, min_epoch_count)

fig, ax = plt.subplots(1, 1, figsize=(right_most_boundary//10, 5))

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
accuracies_ = sorted(accuracies.items(), key=lambda x: task_ids[x[0]])
for i, (task_name, acc) in  enumerate(accuracies_):
    try:
        labels.append(task_name)
    except:
        labels.append('combined')
        
    matrix.append(acc[:right_most_boundary])

plt.sca(ax)

matrix = np.asarray(matrix)
print(matrix.shape)
im = ax.imshow(np.asarray(matrix, dtype=np.float), cmap='autumn', aspect=1)
print(labels)
plt.yticks(range(len(labels)),labels)

fig.colorbar(im, orientation='vertical', aspect=2)

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
fig.suptitle('Combined Accuracy', fontsize=18, ha='center')

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
plt.savefig('combined_accuracy_heatmap.png', bbox_inches='tight')
plt.show()
