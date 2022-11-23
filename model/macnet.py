import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.log = logging.getLogger(__file__)
        self.size_log = logging.getLogger(__file__ + '.size')
        self.size_log.info('size_log')
        self.log.setLevel(logging.CRITICAL)
        self.size_log.setLevel(logging.CRITICAL)
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
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_control = nn.Linear( 2 * hidden_dim, hidden_dim)
        self.attend = nn.Linear(hidden_dim, 1)

        
    def forward(self, prev_control, prev_query, query_repr, memory, mask):
        #import pdb; pdb.set_trace()
        cqi = self.__( self.query_control(
            torch.cat([prev_control, prev_query], dim=-1)), 'cqi')
            
        cais = self.__( self.attend(cqi * query_repr), 'cais')
        cvis = self.__( torch.softmax(cais, dim=0), 'cvis')
        ci   = self.__( (cvis * query_repr).sum(dim=0), 'ci')
        return ci, cvis
    
class ReadUnit(Base):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.project_story = nn.Linear(hidden_dim, hidden_dim)
        self.project_memory  = nn.Linear(hidden_dim, hidden_dim)

        self.input_control = nn.Linear(hidden_dim, 1)
        self.attend = nn.Linear(self.hidden_dim, 1)

    def forward(self, memory, control, story, mask):
        seq_len, batch_size, hidden_dim = story.size()
        projected_story = self.__(
            self.project_story(story)
            .view(seq_len, batch_size, hidden_dim),
            'projected_story'
        )
        projected_memory = self.__( self.project_memory(memory), 'projected_memory')
        Iihw  = self.__( projected_memory * projected_story, 'Iihw')
            
        #control = self.__( control.unsqueeze(0).expand_as(Iihw), 'control')
        raihw = self.__(self.input_control(control * Iihw), 'raihw')
        rvihw = self.__(torch.softmax(raihw,dim=0), 'rvihw')
        ri    = self.__((rvihw * story).sum(dim=0), 'ri')
        return ri, rvihw

class WriteUnit(Base):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_info = nn.Linear(2 * hidden_dim, hidden_dim)
        self.project_control = nn.Linear(hidden_dim, 1)
        self.attend = nn.Linear(hidden_dim, 1)
        self.project_memory = nn.Linear(hidden_dim, hidden_dim)
        self.project_memories = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, memory, read_info, control, prev_controls, prev_memories):
        
        prev_controls = self.__( torch.stack(prev_controls), 'prev_controls')
        prev_memories = self.__( torch.stack(prev_memories), 'prev_memories')
        minfo = self.__(
            self.memory_info(torch.cat([memory, read_info], dim=-1)),
            'minfo')

        projected_control = torch.sigmoid(
            self.__(
                self.project_control(control),
                'projected_control')
        )
        
        control_ = self.__( control.unsqueeze(0).expand_as(prev_controls), 'control')
                
        saij = self.__( F.softmax(self.attend(control_ * prev_controls), dim=0), 'saij')
        misa = self.__( (saij * prev_memories).sum(dim=0), 'misa')
        
        projected_memory = self.__( self.project_memory(minfo), 'projected_memory')
        projected_memories = self.__( self.project_memories(misa), 'projected_memories')
        mip = self.__( projected_memory + projected_memories, 'mi prime' )
            
        mi = self.__( projected_control * memory + (1-projected_control) * mip, 'mi')            
        return mi, projected_control
        
        
class MacNet(Base):
    def __init__(self, embed_dim, hidden_dim, input_vocab_size, output_vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size

        
        self.embed = nn.Embedding(input_vocab_size, embed_dim)
        
        self.encode_story  = nn.GRU(embed_dim,
                                    hidden_dim//2,
                                    bidirectional=True)
        
        self.encode_question = nn.GRU(embed_dim,
                                      hidden_dim//2,
                                      bidirectional=True)
            
        self.dropout = nn.Dropout(0.1)

        self.produce_qi = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.control = ControlUnit (hidden_dim)
        self.read    = ReadUnit    (hidden_dim)
        self.write   = WriteUnit   (hidden_dim)

        self.add_module('control', self.control)
        self.add_module('read'   , self.read   )
        self.add_module('write'  , self.write  )

        self.project = nn.Linear(2 * hidden_dim, hidden_dim)
        self.answer = nn.Linear(hidden_dim, output_vocab_size)

        if config.CONFIG.cuda:
             self.cuda()
        
    def forward(self, input_):
        idxs, inputs, targets = input_
        story, question = inputs
        story = self.__( story, 'story')
        question = self.__(question, 'question')

        story_mask = story > 0
        question_mask = question > 0

        batch_size, story_size  = story.size()
        batch_size, question_size = question.size()
        
        story  = self.__( self.embed(story),  'story_emb')
        question = self.__( self.embed(question), 'question_emb')

        
        story  = story.transpose(1,0)
        story_mask  = story_mask.transpose(1,0)
        story, _  = self.__(  self.encode_story(story), 'S')
        
        question  = question.transpose(1,0)
        question_mask  = question_mask.transpose(1,0)
        question, _ = self.__(  self.encode_question(question), 'Q')

        story_mask = story_mask.unsqueeze(-1).expand_as(story).float()
        question_mask = question_mask.unsqueeze(-1).expand_as(question).float()
        
        mask = (story_mask, question_mask)
        
        c, m, r = [], [], []
        c.append(torch.zeros((batch_size, self.hidden_dim)))
        m.append(torch.zeros((batch_size, self.hidden_dim)))


        
        qi = self.dropout(self.produce_qi(torch.cat([question[-1], m[-1]], dim=-1)))

        qattns, sattns, mattns = [], [], []
        
        for i in range(config.HPCONFIG.reasoning_steps):

            ci, qattn = self.control(c[-1], qi, question, m[-1], mask)
            ci = self.dropout(ci)

            ri, sattn = self.read(m[-1], ci, story, mask)
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
            
        #projected_output = self.__( F.relu(self.project(torch.cat([qi, mi], dim=-1))), 'projected_output')
        return (self.__( F.log_softmax(self.answer(mi), dim=-1), 'return val'),
                (
                    torch.stack(sattns),
                    torch.stack(qattns),
                    torch.stack(mattns)
                )
        )
            


