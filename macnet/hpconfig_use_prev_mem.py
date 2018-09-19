import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    tasks = [
        1,    # Single supporting act
        2,    # two supporting facts
        3,    # three supporting facts
        4,    # two arg relations
        5,    # three arg relations 
        6,    # yes-no-questions
        7,    # counting
        8,    # list-sets
        9,    # simple-negation
        10,   # indefinite-knowledge
        11,   # basic-coreference
        12,   # conjuction
        13,   # compound-coreference
        14,   # time-reasoning
        15,   # basic-deduction
        16,   # basic-induction
        17,   # positional-reasoning
        18,   # size-reasoning
        19,   # path-finding
        20,   # agent-motivations
    ] 

    trainset_size = 1.0
    max_story_len = 0
    hidden_size = 100
    embed_size = 100
    num_layers = 1
    reasoning_steps = 2
    
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'

    class MacNet(Base):
        same_rnn = True

        class CU(Base):
            use_prev_memory = True

        class RU(Base):
            use_story_again = True

        class WU(Base):
            graph_reasoning = True
        
