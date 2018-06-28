import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class Config(Base):
    split_ratio = 0.90
    dropout = 0.1
    cuda = True
    tqdm = True
    flush = True
    
    class Log(Base):
        class _default(Base):
            level=logging.CRITICAL
        class PREPROCESS(Base):
            level=logging.DEBUG
        class MODEL(Base):
            level=logging.INFO
        class TRAINER(Base):
            level=logging.INFO
        class DATAFEED(Base):
            level=logging.INFO
