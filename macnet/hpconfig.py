import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class Config(Base):
    hidden_size = 300
    embed_size = 200
    batch_size = 2

    reasoning_steps = 2
