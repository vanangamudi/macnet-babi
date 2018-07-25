import os
import sys
import json
import time
import random
from pprint import pprint, pformat


from anikattu.logger import CMDFilter
import logging
from pprint import pprint, pformat

logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from config import Config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.trainer import Trainer, Feeder, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.utilz import tqdm, ListTable

from functools import partial

from collections import namedtuple, defaultdict
import itertools

from utilz import MacNetSample as Sample
from utilz import PAD,  word_tokenize
from utilz import VOCAB
from anikattu.utilz import pad_seq

from anikattu.utilz import logger
from anikattu.vocab import Vocab
from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np

import glob

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def load_data(max_sample_size=None):
    samples = []
    qn, an = 0, 0
    skipped = 0

    input_vocabulary = defaultdict(int)
    output_vocabulary = defaultdict(int)
    
    try:
        for i, file_ in enumerate(glob.glob('dataset/en-10k/qa*_train.txt')):
            dataset = open(file_).readlines()
            prev_linenum = 1000000
            for line in tqdm(dataset):
                questions, answers = [], []
                linenum, line = line.split(' ', 1)

                linenum = int(linenum)
                if prev_linenum > linenum:
                    story = ''

                if '?' in line:
                    q, a, _ = line.split('\t')

                    samples.append(
                        Sample('{}.{}'.format(i, linenum),
                               i, linenum,
                               TokenString(story, word_tokenize),
                               TokenString(q,     word_tokenize),
                               a)
                        )

                else:
                    story += ' ' + line

                prev_linenum = linenum

    except:
        skipped += 1
        log.exception('{}'.format(i, linenum))
        
    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.story), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building input_vocabulary...')
    for sample in samples:
        for token in sample.story + sample.q:
            input_vocabulary[token] += 1
            
        output_vocabulary[sample.a] += 1

    return samples, input_vocabulary, output_vocabulary


# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (story, question), (answer) = batch
    return loss_function(output, answer)

def accuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    return (output.max(dim=1)[1] == answer).sum().float()/float(answer.size(0))

def repr_function(output, batch, VOCAB, LABELS):
    indices, (story, question), (answer) = batch
    
    results = []
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, q, a, o in zip(indices, story, question, answer, output):

        c = ' '.join([VOCAB[i] for i in c])
        q = ' '.join([VOCAB[i] for i in q])
        a = ' '.join([LABELS[a]])
        o = ' '.join([LABELS[o]])
        
        results.append([ c, q, a, o ])
        
    return results

def batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []
    answer = []

    for d in datapoints:
        story.append([VOCAB[w] for w in d.story] + [VOCAB['EOS']])
        question.append([VOCAB[w] for w in d.q] + [VOCAB['EOS']])
        answer.append(LABELS[d.a])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))
    answer   = LongVar(answer)

    batch = indices, (story, question), (answer)
    return batch

class Base(nn.Module):
    def __init__(self, Config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(logging.INFO)
        self.size_log.setLevel(logging.INFO)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)

                   
class ControlUnit(Base):
    def __init__(self, Config, name):
        super().__init__(Config, name)
        self.hidden_size = Config.hidden_size

        self.query_control = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.attend = nn.Linear(2 * self.hidden_size, 1)


    def forward(self, prev_control, prev_query, query_repr):
        cqi = self.__( self.query_control(torch.cat([prev_control, prev_query], dim=-1)), 'cqi')
        cais = self.__( self.attend(cqi * query_repr), 'cais') 
        cvis = self.__( F.softmax(cais, dim=0), 'cvis')
        ci   = self.__( (cvis * query_repr).sum(dim=0), 'ci')
        return ci
    
class ReadUnit(Base):
    def __init__(self, Config, name):
        super().__init__(Config, name)
        self.hidden_size = Config.hidden_size

        self.project_story = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.project_memory  = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)

        self.blend = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.input_control = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.attend = nn.Linear(self.hidden_size, 1)

    def forward(self, memory, control, story):
        seq_len, batch_size, hidden_size = story.size()
        projected_story = self.__( self.project_story(story).view(seq_len, batch_size, hidden_size), 'projected_story')
        projected_memory = self.__( self.project_memory(memory), 'projected_memory')
        Iihw  = self.__( projected_memory * projected_story, 'Iihw')
        Iihwp = self.__(self.blend(torch.cat([Iihw, story], dim=-1)), 'Iihwp')

        Iihwp_ = self.__( Iihwp.transpose(0,1), 'Iihwp_' )
        control = self.__( control.unsqueeze(1).expand_as(Iihwp_), 'control')
        raihw = self.__(self.input_control(control * Iihwp_), 'raihw')
        rvihw = self.__(F.softmax(raihw, dim=1), 'rvihw')
        ri    = self.__((rvihw.transpose(0,1) * story).sum(dim=0), 'ri')
        return ri

class WriteUnit(Base):
    def __init__(self, Config, name):
        super().__init__(Config, name)
        self.hidden_size = Config.hidden_size

        self.memory_info = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.attend = nn.Linear(2 * self.hidden_size, 1)

        self.project_memory = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.project_memories = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.project_control = nn.Linear(2 * self.hidden_size, 1)
        
    def forward(self, memory, read_info, control, prev_controls, prev_memories):
        
        prev_controls = self.__( torch.stack(prev_controls), 'prev_controls')
        prev_memories = self.__( torch.stack(prev_memories), 'prev_memories')
        
        minfo = self.__( self.memory_info(torch.cat([memory, read_info], dim=-1)), 'minfo')
        control_ = self.__( control.unsqueeze(0).expand_as(prev_controls), 'control')
                
        saij = self.__( F.softmax(self.attend(control_ * prev_controls), dim=0), 'saij')
        misa = self.__( (saij * prev_memories).sum(dim=0), 'misa')

        projected_memory = self.__( self.project_memory(minfo), 'projected_memory')
        projected_memories = self.__( self.project_memories(misa), 'projected_memories')
        mip = self.__( projected_memory + projected_memories, 'mi prime' )

        projected_control = F.sigmoid(self.__(self.project_control(control), 'projected_control'))

        mi = self.__( projected_control * memory + (1-projected_control) * mip, 'mi')

        return mi
        
        
class MacNet(Base):
    def __init__(self, Config, name, input_vocab_size, output_vocab_size):
        super().__init__(Config, name)

        self.embed_size = Config.embed_size
        self.hidden_size = Config.hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_size)
        
        self.encode_story  = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.encode_question = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.produce_qi = []
        for i in range(Config.reasoning_steps) :
            l = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size) 
            self.add_module('produce_qi:{}'.format(i), l)
            self.produce_qi.append(l)
        
        self.control = ControlUnit (Config, self.name('control'))
        self.read    = ReadUnit    (Config, self.name('read'))
        self.write   = WriteUnit   (Config, self.name('write'))

        self.add_module('control', self.control)
        self.add_module('read'   , self.read   )
        self.add_module('write'  , self.write  )

        self.project = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.answer = nn.Linear(self.hidden_size, self.output_vocab_size)
        
    def forward(self, input_):
        idxs, inputs, targets = input_
        story, question = inputs
        story = self.__( story, 'story')
        question = self.__(question, 'question')

        batch_size, story_size  = story.size()
        batch_size, question_size = question.size()
        
        story  = self.__( self.embed(story),  'story_emb')
        question = self.__( self.embed(question), 'question_emb')

        story  = story.transpose(1,0)
        story, _  = self.__(  self.encode_story(
            story,
            init_hidden(batch_size, self.encode_story)), 'C'
        )
        
        question  = question.transpose(1,0)
        question, _ = self.__(  self.encode_question(
            question,
            init_hidden(batch_size, self.encode_question)), 'Q'
        )

        c, m, r = [], [], []
        c.append(Var(np.zeros((batch_size, 2 * self.hidden_size))))
        m.append(Var(np.zeros((batch_size, 2 * self.hidden_size))))
        for i in range(Config.reasoning_steps):
            qi = self.dropout(self.produce_qi[i](question[-1]))
            ci = self.dropout(self.control(c[-1], qi, question))
            ri = self.dropout(self.read(m[-1], ci, story))
            mi = self.dropout(self.write( m[-1], ri, ci, c, m ))
            
            c.append(ci)
            r.append(ri)
            m.append(mi)
            
        projected_output = self.__( F.relu(self.project(torch.cat([qi, mi], dim=-1))), 'projected_output')
        return self.__( F.log_softmax(self.answer(projected_output), dim=-1), 'return val')
            
def experiment(VOCAB, LABELS, datapoints=[[], [], []], eons=1000, epochs=10, checkpoint=1):
    try:
        try:
            model =  MacNet(Config(), 'macnet', len(VOCAB),  len(LABELS))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, 'pth')))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')

        print('**** the model', model)

        name = SELF_NAME
        _batchop = partial(batchop, VOCAB=VOCAB, LABELS=LABELS)
        train_feed     = DataFeed(name, datapoints[0], batchop=_batchop, batch_size=32)
        test_feed      = DataFeed(name, datapoints[1], batchop=_batchop, batch_size=32)
        predictor_feed = DataFeed(name, datapoints[2], batchop=_batchop, batch_size=1)

        loss_ = partial(loss, loss_function=nn.NLLLoss())
        trainer = Trainer(name=name,
                          model=model, 
                          loss_function=loss_, accuracy_function=accuracy, 
                          checkpoint=checkpoint, epochs=epochs,
                          feeder = Feeder(train_feed, test_feed))

        predictor = Predictor(model=model, feed=predictor_feed, repr_function=partial(repr_function, VOCAB=VOCAB, LABELS=LABELS))
        
        for e in range(eons):            
            dump = open('results/{}/eon_{}.csv'.format(SELF_NAME, e), 'a')
            log.info('on {}th eon'.format(e))
            results = ListTable()
            for ri in tqdm(range(predictor_feed.num_batch)):
                output, _results = predictor.predict(ri)
                results.extend(_results)
            dump.write(repr(results))
            dump.close()
            
            if not trainer.train():
                raise Exception

        
    except KeyboardInterrupt:
        trainer.save_best_model()
        return locals()
    except :
        log.exception('####################')
        return locals()
    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    if Config.flush:
        log.info('flushing...')
        dataset, vocabulary, labels = load_data()
        pickle.dump([dataset, dict(vocabulary), dict(labels)], open('train.babi', 'wb'))
    else:
        dataset, _vocabulary, _labels = pickle.load(open('train.babi', 'rb'))
        vocabulary = defaultdict(int); labels = defaultdict(int)
        vocabulary.update(_vocabulary), labels.update(_labels)
        
    log.info('dataset size: {}'.format(len(dataset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset[0])))

    log.info('vocabulary: {}'.format(
        pformat(
            sorted(
                vocabulary.items(), key=lambda x: x[1], reverse=True)
        )))

    log.info(pformat(labels))
    VOCAB  = Vocab(vocabulary, VOCAB)
    LABELS = Vocab(labels)
    
    if 'train' in sys.argv:
        labelled_samples = [d for d in dataset if len(d.a) > 0] #[:100]
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: -len(x.story))
        test_set  = sorted(test_set, key=lambda x: -len(x.story))
        exp_image = experiment(VOCAB, LABELS, datapoints=[train_set, test_set, test_set])
        
    if 'predict' in sys.argv:
        model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
        if Config().cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, '.pth')))
        start_time = time.time()
        strings = sys.argv[2]
        
        s = [WORD2INDEX[i] for i in word_tokenize(strings)] + [WORD2INDEX['PAD']]
        e1, e2 = [WORD2INDEX['ENTITY1']], [WORD2INDEX['ENTITY2']]
        output = model(s, e1, e2)
        output = output.data.max(dim=-1)[1].cpu().numpy()
        label = LABELS[output[0]]
        print(label)

        duration = time.time() - start_time
        print(duration)
