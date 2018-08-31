import os
import re
import sys
import glob
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.trainer import Trainer, Tester, Predictor
from anikattu.trainer.multiplexed_trainer import MultiplexedTrainer
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
MacNetSample   =  namedtuple('MacNetSample', ['id', 'aid', 'qid', 'task_name', 'story', 'q', 'a'])

def load_task_data(config, task=1, type_='train', max_sample_size=None):
    samples = []
    qn, an = 0, 0
    skipped = 0

    input_vocabulary = Counter()
    output_vocabulary = Counter()
    
    try:
        filename = glob.glob('../dataset/en-10k/qa{}_*_{}.txt'.format(task, type_))[0]
              
        task_name = re.search(r'qa\d+_(.*)_.*.txt', filename)
        if task_name:
            task_name = task_name.group(1)
            
        log.info('processing file: {}'.format(filename))
        dataset = open(filename).readlines()
        prev_linenum = 1000000
        for line in tqdm(dataset, desc='processing {}'.format(filename)):
            questions, answers = [], []
            linenum, line = line.split(' ', 1)

            linenum = int(linenum)
            if prev_linenum > linenum:
                story = ''

            if '?' in line:
                q, a, _ = line.split('\t')

                if config.max_story_len:
                    if config.max_story_len < len(word_tokenize(story)):
                        continue
                    
                samples.append(
                    MacNetSample('{}.{}'.format(task, linenum),
                           task, linenum,
                           task_name,
                           TokenString(story.lower(), word_tokenize),
                           TokenString(q.lower(),     word_tokenize),
                           a.lower())
                    )


                if  max_sample_size and len(samples) > max_sample_size:
                    break

            else:
                story += ' ' + line

            prev_linenum = linenum

    except:
        skipped += 1
        log.exception('{}'.format(task, linenum))

        
        
    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.story), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building input_vocabulary...')
    for sample in samples:
        input_vocabulary.update(sample.story + sample.q)            
        output_vocabulary.update([sample.a])

    return filename, samples, input_vocabulary, output_vocabulary



def load_data(config, max_sample_size=None):
    dataset = {}
    for i in config.HPCONFIG.tasks:
        filename, train_samples, train_input_vocab, train_output_vocab = load_task_data(config, task=i, type_='train', max_sample_size=max_sample_size)
        filename, test_samples, test_input_vocab, test_output_vocab = load_task_data(config, task=i, type_='test', max_sample_size=max_sample_size)
        task_name = re.search(r'qa\d+_(.*)_.*.txt', filename)

        if task_name:
            task_name = task_name.group(1)

        input_vocab = train_input_vocab + test_input_vocab
        output_vocab = train_output_vocab + test_output_vocab
        dataset[task_name] = Dataset(task_name, (train_samples, test_samples), Vocab(input_vocab, special_tokens=VOCAB), Vocab(output_vocab))

    return DatasetList('babi', dataset.values())
        

# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, attn = output
    return loss_function(output, answer)

def accuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, attn = output
    return (output.max(dim=1)[1] == answer).sum().float()/float(answer.size(0))

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, (story, question), (answer) = batch
    output, attn = output
    results = []
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, q, a, o in zip(indices, story, question, answer, output):

        c = ' '.join([VOCAB[i] for i in c]).replace('\n', ' ')
        q = ' '.join([VOCAB[i] for i in q])
        a = ' '.join([LABELS[a]])
        o = ' '.join([LABELS[o]])
        
        results.append([idx, dataset[idx].task_name, c, q, a, o, str(a == o) ])
        
    return results


def batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []
    answer = []

    for d in datapoints:
        story.append([VOCAB[w] for w in d.story])
        question.append([VOCAB[w] for w in d.q])
        answer.append(LABELS[d.a])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))
    answer   = LongVar(answer)

    batch = indices, (story, question), (answer)
    return batch


def predict_batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []

    for d in datapoints:
        story.append([VOCAB[w] for w in d.story])
        question.append([VOCAB[w] for w in d.q])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))

    batch = indices, (story, question), ()
    return batch

def portion(dataset, percent):
    return dataset[ : int(len(dataset) * percent) ]

def train(config, argv, name, ROOT_DIR,  model, dataset):
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab)
    predictor_feed = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=1)
    train_feed     = DataFeed(name, portion(dataset.trainset, config.HPCONFIG.trainset_size),
                              batchop=_batchop, batch_size=config.CONFIG.batch_size)
    
    predictor = Predictor(name,
                          model=model,
                          directory=ROOT_DIR,
                          feed=predictor_feed,
                          repr_function=partial(repr_function
                                                , VOCAB=dataset.input_vocab
                                                , LABELS=dataset.output_vocab
                                                , dataset=dataset.testset_dict))

    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed, tester = {}, {}
    for subset in dataset.datasets:
        test_feed[subset.name]      = DataFeed(subset.name, subset.testset, batchop=_batchop, batch_size=config.CONFIG.batch_size)

        tester[subset.name] = Tester(name     = subset.name,
                                     config   = config,
                                     model    = model,
                                     directory = ROOT_DIR,
                                     loss_function = loss_,
                                     accuracy_function = accuracy,
                                     feed = test_feed[subset.name],
                                     save_model_weights=False)

    test_feed[name]      = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=config.CONFIG.batch_size)

    tester[name] = Tester(name  = name,
                                  config   = config,
                                  model    = model,
                                  directory = ROOT_DIR,
                                  loss_function = loss_,
                                  accuracy_function = accuracy,
                                  feed = test_feed[name],
                                  predictor=predictor)


    def do_every_checkpoint(epoch):
        if config.CONFIG.plot_metrics:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10, 5))
            
        for t in tester.values():
            t.do_every_checkpoint(epoch)

            if config.CONFIG.plot_metrics:
                plt.plot(list(t.accuracy), label=t.name)

        if config.CONFIG.plot_metrics:
            plt.savefig('accuracy.png')
            plt.close()
        


    trainer = Trainer(name=name,
                      config = config,
                      model=model,
                      directory=ROOT_DIR,
                      optimizer  = optim.Adam(model.parameters()),
                      loss_function = loss_,
                      checkpoint = config.CONFIG.CHECKPOINT,
                      do_every_checkpoint = do_every_checkpoint,
                      epochs = config.CONFIG.EPOCHS,
                      feed = train_feed,
    )



    for e in range(config.CONFIG.EONS):

        if not trainer.train():
            raise Exception

        dump = open('{}/results/eon_{}.csv'.format(ROOT_DIR, e), 'w')
        log.info('on {}th eon'.format(e))
        results = ListTable()
        for ri in tqdm(range(predictor_feed.num_batch), desc='running prediction on eon: {}'.format(e)):
            output, _results = predictor.predict(ri)
            results.extend(_results)
        dump.write(repr(results))
        dump.close()

def multiplexed_train(config, argv, name, ROOT_DIR,  model, dataset):
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab)
    predictor_feed = DataFeed(name, dataset.testset, batchop=_batchop, batch_size=1)
    predictor = Predictor(name,
                          model=model,
                          directory=ROOT_DIR,
                          feed=predictor_feed,
                          repr_function=partial(repr_function
                                                , VOCAB=dataset.input_vocab
                                                , LABELS=dataset.output_vocab
                                                , dataset=dataset.testset_dict))

    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed, tester = {}, {}
    train_feed = {}
    for subset in dataset.datasets:
        test_feed[subset.name]      = DataFeed(subset.name, subset.testset,
                                               batchop=_batchop, batch_size=config.CONFIG.batch_size)
        train_feed[subset.name]     = DataFeed(subset.name,
                                               portion(subset.trainset, config.HPCONFIG.trainset_size),
                                               batchop=_batchop, batch_size=config.CONFIG.batch_size)
    
        tester[subset.name] = Tester(name     = subset.name,
                                     config   = config,
                                     model    = model,
                                     directory = ROOT_DIR,
                                     loss_function = loss_,
                                     accuracy_function = accuracy,
                                     feed = test_feed[subset.name],
                                     save_model_weights=False)

    test_feed[name]      = DataFeed(name, dataset.testset,
                                    batchop=_batchop, batch_size=config.CONFIG.batch_size)

    tester[name] = Tester(name  = name,
                                  config   = config,
                                  model    = model,
                                  directory = ROOT_DIR,
                                  loss_function = loss_,
                                  accuracy_function = accuracy,
                                  feed = test_feed[name],
                                  predictor=predictor)

    train_feed_muxed = MultiplexedDataFeed(name,
                                           train_feed,
                                           _batchop,
                                           config.CONFIG.batch_size)
    trainer = MultiplexedTrainer(name=name,
                                 config = config,
                                 model=model,
                                 directory=ROOT_DIR,
                                 optimizer  = optim.Adam(model.parameters()),
                                 loss_function = loss_,
                                 testers=tester,
                                 checkpoint = config.CONFIG.CHECKPOINT,
                                 epochs = config.CONFIG.EPOCHS,
                                 feed = train_feed_muxed,
    )



    for e in range(config.CONFIG.EONS):

        if not trainer.train():
            raise Exception

        dump = open('{}/results/eon_{}.csv'.format(ROOT_DIR, e), 'w')
        log.info('on {}th eon'.format(e))
        results = ListTable()
        for ri in tqdm(range(predictor_feed.num_batch), desc='\nrunning prediction on eon: {}'.format(e)):
            output, _results = predictor.predict(ri)
            results.extend(_results)
        dump.write(repr(results))
        dump.close()

    
def predict(config, argv, task, model, input_string, dataset):                
    story, question = input_string.lower().split('|') 
    story_tokens = word_tokenize(story)
    question_tokens = word_tokenize(question)
    
    input_ = predict_batchop(
        datapoints = [MacNetSample('0', 'story 1', 'question 1', 'task 1',
                             story_tokens, question_tokens, '')],
        VOCAB      = dataset.input_vocab,
        LABELS     = dataset.output_vocab
    )
            
    output = model(input_)
    plot_attn1(config, argv, task, question_tokens, story_tokens, output, dataset)
    
def plot_attn1(config, argv, task, question_tokens, story_tokens, output, dataset):
    output, (sattn, qattn, mattn) = output
    sattn = sattn.squeeze().data.cpu()
    qattn = qattn.squeeze().data.cpu()
    print('story_tokens', len(story_tokens))

    if sattn.dim() == 1: sattn = sattn.unsqueeze(0)
    if qattn.dim() == 1: qattn = qattn.unsqueeze(0)
    print('sattn', sattn.size())
    print('qattn', qattn.size())

    #story_tokens += ['EOS']
    #question_tokens += ['EOS']

    answer = dataset.output_vocab[output.max(1)[1]]
    print(answer)
    if 'show_plot' in argv or 'save_plot' in argv:
        from matplotlib import pyplot as plt
        plt.style.use('ggplot')
        fig, axes = plt.subplots(sattn.size(0), 2,
                                 figsize=( max(2, sattn.size(1)),  max(16, sattn.size(0))),
                                 gridspec_kw = {
                                     'width_ratios': [
                                         len(question_tokens),
                                         len(story_tokens)]
                                 }
        )

        if axes.ndim == 1: axes = axes.reshape(1, -1)
        #if type(axes[0]) != list: axes = [axes]
        plt.suptitle('{}\n{}'.format(' '.join(question_tokens), answer))
        for i, ax in enumerate(axes):
            ax1, ax2 = ax
            for j in ax:
                j.set_ylim(0, 1)
            #ax1.set_aspect(30 * 1/len(question_tokens), adjustable='box')
            #ax2.set_aspect(20 * 1/len(question_tokens), adjustable='box')

            nwords = len(question_tokens)
            plt.sca(ax1)
            plt.bar(range(nwords), qattn[i].tolist())
            plt.xticks(range(nwords), question_tokens, rotation='vertical')

            nwords = len(story_tokens)
            plt.sca(ax2)
            plt.bar(range(nwords), sattn[i].tolist())
            plt.xticks(range(nwords), story_tokens, rotation='vertical')
            
        
        if 'save_plot' in argv:
            plt.savefig('{}/plots/{}/{}.png'.format(config.ROOT_DIR, task,  '_'.join(question_tokens)))
            
        if 'show_plot' in argv:
            plt.show()
        plt.close()
    
