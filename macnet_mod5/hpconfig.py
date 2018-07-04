import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    hidden_size = 100
    embed_size = 100
    batch_size = 200
    reasoning_steps = 1
    LR = 0.001
    MOMENTUM=0.1
