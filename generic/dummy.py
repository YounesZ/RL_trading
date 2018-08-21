import numpy as np


class Model():

    def __init__(self, type):
        self.qmodel     =   type


class Noise():

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return 0