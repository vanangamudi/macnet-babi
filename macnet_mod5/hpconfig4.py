import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    trainset_size = 1.0
    hidden_size = 20
    embed_size = 20
    num_layers = 1
    reasoning_steps = 2
    
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'
