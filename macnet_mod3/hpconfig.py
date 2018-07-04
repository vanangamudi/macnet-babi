import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    hidden_size = 200
    embed_size = 200
    batch_size = 8
    reasoning_steps = 8
    LR = 0.001
    MOMENTUM=0.1
