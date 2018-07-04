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

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.utilz import initialize_task

from model.macnet_mod2 import MacNet
from utilz import load_data, train, predict


SELF_NAME = os.path.basename(__file__).replace('.py', '')

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
    log.info('dataset[:10]: {}'.format(pformat(dataset.trainset[-1])))

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
        train(config, sys.argv, SELF_NAME, ROOT_DIR, model, dataset)
        
    if 'predict' in sys.argv:
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            input_string = input('?')
            if not input_string:
                continue
            
            predict(config, sys.argv, model, input_string, dataset)            
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