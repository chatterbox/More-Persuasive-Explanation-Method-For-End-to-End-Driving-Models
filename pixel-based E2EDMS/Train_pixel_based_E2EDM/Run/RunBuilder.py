#!/usr/bin/env python
# coding: utf-8

from itertools import product
from collections import namedtuple


class RunBuilder():
    # build sets of parameters that define our runs
    @staticmethod
    def get_runs(params):
        # params is the dictionary
        Run  = namedtuple('Run',params.keys())
        
        # Run as this tuple name and param.keys() is the content insides this tuple
        runs=[]
        # print(*params.values()) # ['3DCNN_Paper'] [0.001] [5] [5] [2]
        '''
        e.g.，product（[1,2],[3,4]） = [1,3],[2,3],[1,4],[2,4]
        '''
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
