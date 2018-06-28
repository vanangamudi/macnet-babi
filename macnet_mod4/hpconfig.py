import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    hidden_size = 50
    embed_size = 50
    batch_size = 32

    reasoning_steps = 2

    LR = 0.001
    MOMENTUM=0.1
