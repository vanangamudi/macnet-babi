import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    hidden_size = 5
    embed_size = 5
    batch_size = 1024

    reasoning_steps = 1
