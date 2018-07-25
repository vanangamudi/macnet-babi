import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')

from anikattu.logger import CMDFilter
import logging
from pprint import pprint, pformat

logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.trainer import Trainer, Tester, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.utilz import initialize_task
from functools import partial

from collections import namedtuple, defaultdict, Counter
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

def load_task_data(task=1, type_='train', max_sample_size=None):
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
        for line in tqdm(dataset):
            questions, answers = [], []
            linenum, line = line.split(' ', 1)

            linenum = int(linenum)
            if prev_linenum > linenum:
                story = ''

            if '?' in line:
                q, a, _ = line.split('\t')

                samples.append(
                    Sample('{}.{}'.format(task, linenum),
                           task, linenum,
                           task_name,
                           story.lower(),
                           q.lower(),     
                           a.lower()
                    )
                )

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

def load_data(max_sample_size=None):
    dataset = {}
    for i in range(1, 21):
        filename, train_samples, train_input_vocab, train_output_vocab = load_task_data(task=i, type_='train')
        filename, test_samples, test_input_vocab, test_output_vocab = load_task_data(task=i, type_='test')

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
    return loss_function(output, answer)

def accuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    return (output.max(dim=1)[1] == answer).sum().float()/float(answer.size(0))


def waccuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    index = answer
    src = Var(torch.ones(answer.size()))
    
    acc_nomin = Var(torch.zeros(output.size(1)))
    acc_denom = Var(torch.ones(output.size(1)))

    acc_denom.scatter_add_(0, index, (answer == answer).float() )
    acc_nomin.scatter_add_(0, index, (answer == output.max(1)[1]).float())

    accuracy = acc_nomin / acc_denom

    #pdb.set_trace()
    return accuracy.mean()

def f1score(output, batch, *args, **kwargs):
    indices, (story, question), (target) = batch
    batch_size, class_size  = output.size()

    tp, tn, fp, fn = Var([0]), Var([0]), Var([0]), Var([0])
    p, r, f1 = Var([0]), Var([0]), Var([0])

    i = output
    t = target
    i = i.max(dim=1)[1]
    log.debug('output:{}'.format(pformat(i)))
    log.debug('target:{}'.format(pformat(t)))
    i_ = i
    t_ = t
    tp_ = ( i_ * t_ ).sum().float()
    fp_ = ( i_ > t_ ).sum().float()
    fn_ = ( i_ < t_ ).sum().float()

    i_ = i == 0
    t_ = t == 0
    tn_ = ( i_ * t_ ).sum().float()

    tp += tp_
    tn += tn_
    fp += fp_
    fn += fn_

    log.debug('tp_: {}\n fp_:{} \n fn_: {}\n tn_: {}'.format(tp_, fp_, fn_, tn_))


    if tp_.data.item() > 0:
        p_ = tp_ / (tp_ + fp_)
        r_ = tp_ / (tp_ + fn_)
        f1 += 2 * p_ * r_/ (p_ + r_)
        p += p_
        r += r_

    return (tp, fn, fp, tn), (p), (r), (f1)

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, (story, question), (answer) = batch
    
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
        story.append([VOCAB[w] for w in d.story] + [VOCAB['EOS']])
        question.append([VOCAB[w] for w in d.q] + [VOCAB['EOS']])
        answer.append(LABELS[d.a])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))
    answer   = LongVar(answer)

    batch = indices, (story, question), (answer)
    return batch

class Base(nn.Module):
    def __init__(self, config, name):
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
    def __init__(self, config, name):
        super().__init__(config, name)
        self.hidden_size = config.HPCONFIG.hidden_size

        self.query_control = nn.Linear(6 * self.hidden_size, 2 * self.hidden_size)
        self.attend = nn.Linear(2 * self.hidden_size, 1)


    def forward(self, prev_control, prev_query, query_repr, memory):
        cqi = self.__( self.query_control(torch.cat([prev_control, prev_query, memory], dim=-1)), 'cqi')
        cais = self.__( self.attend(cqi * query_repr), 'cais') 
        cvis = self.__( F.softmax(cais, dim=0), 'cvis')
        ci   = self.__( (cvis * query_repr).sum(dim=0), 'ci')
        return ci
    
class ReadUnit(Base):
    def __init__(self, config, name):
        super().__init__(config, name)
        self.hidden_size = config.HPCONFIG.hidden_size

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
    def __init__(self, config, name):
        super().__init__(config, name)
        self.hidden_size = config.HPCONFIG.hidden_size

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
    def __init__(self, config, name, input_vocab_size, output_vocab_size):
        super().__init__(config, name)
        self.config = config
        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_size)
        
        self.encode_story  = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.encode_question = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.produce_qi = nn.Linear(4*self.hidden_size, 2 * self.hidden_size)
        
        self.control = ControlUnit (config, self.name('control'))
        self.read    = ReadUnit    (config, self.name('read'))
        self.write   = WriteUnit   (config, self.name('write'))

        self.add_module('control', self.control)
        self.add_module('read'   , self.read   )
        self.add_module('write'  , self.write  )

        self.project = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.answer = nn.Linear(self.hidden_size, self.output_vocab_size)

        if config.CONFIG.cuda:
             self.cuda()
        
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
        qi = self.dropout(self.produce_qi(torch.cat([question[-1], m[-1]], dim=-1)))
        for i in range(config.HPCONFIG.reasoning_steps):

            ci = self.dropout(self.control(c[-1], qi, question, m[-1]))
            ri = self.dropout(self.read(m[-1], ci, story))
            mi = self.dropout(self.write( m[-1], ri, ci, c, m ))
            qi = self.dropout(self.produce_qi(torch.cat([qi, m[-1]], dim=-1)))
            
            c.append(ci)
            r.append(ri)
            m.append(mi)
            
        projected_output = self.__( F.relu(self.project(torch.cat([qi, mi], dim=-1))), 'projected_output')
        return self.__( F.log_softmax(self.answer(projected_output), dim=-1), 'return val')
            
    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    ROOT_DIR = initialize_task(SELF_NAME)

    print('====================================')
    print(ROOT_DIR)
    print('====================================')
        
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data()
        pickle.dump(dataset, open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset.trainset[0])))

    log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    
    try:
        model =  MacNet(config, 'macnet', len(dataset.input_vocab),  len(dataset.output_vocab))
        model_snapshot = '{}/weights/{}.{}'.format(ROOT_DIR, SELF_NAME, 'pth')
        model.load_state_dict(torch.load(model_snapshot))
        log.info('loaded the old image for the model from :{}'.format(model_snapshot))
    except:
        log.exception('failed to load the model  from :{}'.format(model_snapshot))
        
    if config.CONFIG.cuda:  model = model.cuda()        
    print('**** the model', model)
    
    if 'train' in sys.argv:
        _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab)
        train_feed     = DataFeed(SELF_NAME, dataset.trainset, batchop=_batchop, batch_size=config.HPCONFIG.batch_size)
        predictor_feed = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=1)

        predictor = Predictor(SELF_NAME,
                              model=model,
                              directory=ROOT_DIR,
                              feed=predictor_feed,
                              repr_function = partial(
                                  repr_function,
                                  VOCAB=dataset.input_vocab,
                                  LABELS=dataset.output_vocab,
                                  dataset=dataset.testset_dict))

        
        loss_ = partial(loss, loss_function=nn.NLLLoss())
        test_feed, tester = {}, {}
        for subset in dataset.datasets:
            test_feed[subset.name]      = DataFeed(subset.name, subset.testset, batchop=_batchop, batch_size=config.HPCONFIG.batch_size)

            tester[subset.name] = Tester(name     = subset.name,
                                         config   = config,
                                         model    = model,
                                         directory = ROOT_DIR,
                                         loss_function = loss_,
                                         accuracy_function = accuracy,
                                         feed = test_feed[subset.name],
                                         save_model_weights=False)
            
        test_feed[SELF_NAME]      = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=config.HPCONFIG.batch_size)

        tester[SELF_NAME] = Tester(name  = SELF_NAME,
                                      config   = config,
                                      model    = model,
                                      directory = ROOT_DIR,
                                      loss_function = loss_,
                                      accuracy_function = accuracy,
                                      feed = test_feed[SELF_NAME],
                                      predictor=predictor)

            
        def do_every_checkpoint(epoch):
            if epoch % 20 == 0:
                for t in tester.values():
                    t.do_every_checkpoint(epoch)
            else:
                tester[SELF_NAME].do_every_checkpoint(epoch)
                    
        trainer = Trainer(name=SELF_NAME,
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
            for ri in tqdm(range(predictor_feed.num_batch)):
                output, _results = predictor.predict(ri)
                results.extend(_results)
            dump.write(repr(results))
            dump.close()

    if 'predict' in sys.argv:
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            sentence = []
            input_string = word_tokenize(input('?').lower())
            sentence.append([VOCAB[w] for w in input_string] + [VOCAB['EOS']])
            dummy_label = LongVar([0])
            sentence = LongVar(sentence)
            input_ = [0], (sentence,), (0, )
            output, attn = model(input_)

            print(LABELS[output.max(1)[1]])

            if 'show_plot' in sys.argv or 'save_plot' in sys.argv:
                nwords = len(input_string)

                from matplotlib import pyplot as plt
                plt.figure(figsize=(20,10))
                plt.bar(range(nwords+1), attn.squeeze().data.cpu().numpy())
                plt.title('{}\n{}'.format(output.exp().tolist(), LABELS[output.max(1)[1]]))
                plt.xticks(range(nwords), input_string, rotation='vertical')
                if 'show_plot' in sys.argv:
                    plt.show()
                if 'save_plot' in sys.argv:
                    plt.savefig('{}.png'.format(count))
                plt.close()

            print('Done')
                
    if 'service' in sys.argv:
        model.eval()
        from flask import Flask,request,jsonify
        from flask_cors import CORS
        app = Flask(__name__)
        CORS(app)

        @app.route('/ade-genentech',methods=['POST'])
        def _predict():
           print(' requests incoming..')
           sentence = []
           try:
               input_string = word_tokenize(request.json["text"].lower())
               sentence.append([VOCAB[w] for w in input_string] + [VOCAB['EOS']])
               dummy_label = LongVar([0])
               sentence = LongVar(sentence)
               input_ = [0], (sentence,), (0, )
               output, attn = model(input_)
               #print(LABELS[output.max(1)[1]], attn)
               nwords = len(input_string)
               return jsonify({
                   "result": {
                       'sentence': input_string,
                       'attn': ['{:0.4f}'.format(i) for i in attn.squeeze().data.cpu().numpy().tolist()[:-1]],
                       'probs': ['{:0.4f}'.format(i) for i in output.exp().squeeze().data.cpu().numpy().tolist()],
                       'label': LABELS[output.max(1)[1].squeeze().data.cpu().numpy()]
                   }
               })
           
           except Exception as e:
               print(e)
               return jsonify({"result":"model failed"})

        print('model running on port:5010')
        app.run(host='0.0.0.0',port=5010)
