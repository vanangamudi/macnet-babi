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
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


import config
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden

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
        return ci, cvis
    
class ReadUnit(Base):
    def __init__(self, config, name):
        super().__init__(config, name)
        self.hidden_size = config.HPCONFIG.hidden_size

        self.project_story = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.project_memory  = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)

        self.blend = nn.Linear(4 * self.hidden_size, 2 * self.hidden_size)
        self.input_control = nn.Linear(2 * self.hidden_size, 1)
        self.attend = nn.Linear(self.hidden_size, 1)

    def forward(self, memory, control, story):
        seq_len, batch_size, hidden_size = story.size()
        projected_story = self.__( self.project_story(story).view(seq_len, batch_size, hidden_size), 'projected_story')
        projected_memory = self.__( self.project_memory(memory), 'projected_memory')
        Iihw  = self.__( projected_memory * projected_story, 'Iihw')
        Iihwp = self.__(self.blend(torch.cat([Iihw, story], dim=-1)), 'Iihwp')

        Iihwp_ = self.__( Iihwp, 'Iihwp_' )
        control = self.__( control.unsqueeze(0).expand_as(Iihwp_), 'control')
        raihw = self.__(self.input_control(control * Iihwp_), 'raihw')
        rvihw = self.__(F.softmax(raihw, dim=0), 'rvihw')
        ri    = self.__((rvihw * story).sum(dim=0), 'ri')
        return ri, rvihw

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

        return mi, projected_control
        
        
class MacNet(Base):
    def __init__(self, config, name, input_vocab_size, output_vocab_size):
        super().__init__(config, name)
        self.config = config
        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_size)
        
        self.encode_story  = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True, num_layers=config.HPCONFIG.num_layers)
        self.encode_question = nn.GRU(self.embed.embedding_dim, self.hidden_size, bidirectional=True, num_layers=config.HPCONFIG.num_layers)
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

        qattns, sattns, mattns = [], [], []
        
        for i in range(config.HPCONFIG.reasoning_steps):

            ci, qattn = self.control(c[-1], qi, question, m[-1])
            ci = self.dropout(ci)

            ri, sattn = self.read(m[-1], ci, story)
            ri = self.dropout(ri)

            mi, mattn = self.write( m[-1], ri, ci, c, m )
            mi = self.dropout(mi)

            qi = self.dropout(self.produce_qi(torch.cat([qi, m[-1]], dim=-1)))
            
            c.append(ci)
            r.append(ri)
            m.append(mi)

            qattns.append(qattn)
            sattns.append(sattn)
            mattns.append(mattn)
            
        projected_output = self.__( F.relu(self.project(torch.cat([qi, mi], dim=-1))), 'projected_output')
        return (self.__( F.log_softmax(self.answer(projected_output), dim=-1), 'return val'),
                (
                    torch.stack(sattns),
                    torch.stack(qattns),
                    torch.stack(mattns)
                )
        )
            


