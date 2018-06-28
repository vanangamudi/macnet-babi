import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    hidden_size = 100
    embed_size = 100
    batch_size = 32

    reasoning_steps = 16
