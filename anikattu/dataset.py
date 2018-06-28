from config import CONFIG
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import random

from .utilz import tqdm
from .debug import memory_consumed
from .vocab import Vocab

from collections import Counter
import random

def split_dataset(dataset, ratio=0.8):
    pivot = ratio * len(dataset)
    return dataset[:pivot], dataset[pivot:]

class NLPDatasetList:
    def __init__(self, name, datasets):
        self.name = name
        
        self.datasets = list(datasets)

        self.trainset, self.testset = [], []
        for dataset in self.datasets:
            self.trainset.extend(dataset.trainset)
            self.testset.extend(dataset.testset)

        self.trainset_dict = {i.id:i for i in self.trainset}
        self.testset_dict = {i.id:i for i in self.testset}
            
        random.shuffle(self.trainset)
        random.shuffle(self.testset)

        self.trainset = sorted(self.trainset, key=lambda x: x.story, reverse=True)
        self.testset = sorted(self.testset, key=lambda x: x.story, reverse=True)
        
        input_vocab = Counter()
        special_tokens = []
        for dataset in self.datasets:
            input_vocab += dataset.input_vocab.freq_dict

            for token in dataset.input_vocab.special_tokens:
                if token not in special_tokens:
                    special_tokens.append(token)

        self.input_vocab = Vocab(input_vocab, special_tokens)        

        output_vocab = Counter()
        special_tokens = []
        for dataset in self.datasets:
            output_vocab += dataset.output_vocab.freq_dict
            special_tokens.extend(dataset.output_vocab.special_tokens)
            
        self.output_vocab = Vocab(output_vocab, special_tokens)

        log.info('build dataset: {}'.format(name))
        log.info(' trainset size: {}'.format(len(self.trainset)))
        log.info(' testset size: {}'.format(len(self.testset)))
        log.info(' input_vocab size: {}'.format(len(self.input_vocab)))
        log.info(' output_vocab size: {}'.format(len(self.output_vocab)))
        
        
    def __iter__(self):
        self.datasets.__iter__()


        
class NLPDataset:
    def __init__(self, name, dataset, input_vocab, output_vocab):

        self.name = name
        log.info('building dataset: {}'.format(name))
        if not isinstance(dataset, tuple):
            dataset =  split_dataset(list(dataset))
        self.trainset, self.testset =  dataset
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab


        self.trainset_dict = {i.id:i for i in self.trainset}
        self.testset_dict = {i.id:i for i in self.testset}
        
        log.info('build dataset: {}'.format(name))
        log.info(' trainset size: {}'.format(len(self.trainset)))
        log.info(' testset size: {}'.format(len(self.testset)))
        log.info(' input_vocab size: {}'.format(len(self.input_vocab)))
        log.info(' output_vocab size: {}'.format(len(self.output_vocab)))
        
