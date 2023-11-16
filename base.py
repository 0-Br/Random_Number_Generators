from pickle import dump, load
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from time import time
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
import QuantLib as ql


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SKIP_PATHS = 2 ** 18 - 1 # quantlib库序列所用的路径跳跃参数
MAX_BATCH_SIZE = 16384 # 最大单次采样规模
STATS_SIZE = 65536 # 统计采样规模

BASE_SCALE = 1000000 # 数据规模

def prevp2(n:int) -> int:
    '''返回不大于n的最大的2的幂次'''
    ex = 0
    while n > 1:
        n >>= 1
        ex += 1
    return 1 << ex


class SeqM(Enum):

    ORIGIN = 'origin'
    DUAL = 'dual'
    STANDARD ='standard'


class Distribution(ABC):

    __objects__ = {} # 缓存
    SAMPLE_RATIO = 1 # 生成时的采样比例

    @classmethod
    def save(cls):
        with open("cache\\(%d)%s" % (BASE_SCALE, cls.__name__), "wb") as f:
            dump(cls.__objects__, f)

    @classmethod
    def read(cls, scale:int):
        with open("cache\\(%d)%s" % (scale, cls.__name__), "rb") as f:
            cls.__objects__ = load(f)

    @classmethod
    def get_obj(cls, key):
        return cls.__objects__[key]

    def __init__(self):
        self.gen_time = None
        self.gen_num = None
        self.gen_type = None
        self.sampled = None

        self.sequences = {}
        self.mean = {}
        self.var = {}
        self.p_KStest = {} # Kolmogorov–Smirnov test
        self.p_CMtest = {} # Cramér–von Mises test

    def __str__(self):
        return self.sequences.__str__()

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def analyse(self):
        pass

    @abstractmethod
    def report(self):
        pass

    def get_sequence(self, st:SeqM=None):
        return deepcopy(self.sequences[st])
