from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn
from torch.autograd import Variable

from collections import namedtuple, defaultdict


from anikattu.tokenizer import word_tokenize

from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
MacNetSample   =  namedtuple('MacNetSample', ['id', 'aid', 'qid', 'task_name', 'story', 'q', 'a'])
