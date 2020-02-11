import cPickle, os
import numpy as np
from utils import deltaPsi
import healpy

class LocalWarmSpotList(object):
    def __init__(self, **kwargs):
        self.warm_spot_list = []
        
class LocalWarmSpotExpectation(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, logP):
        pass
        #N_expected = 
        #return N_expected

class HPA_TS_value(object):
    def __init__(self, **kwargs):
        pass

class HPA_TS_Parametrization(object):
    def __init__(self, **kwargs):
        pass

class SingleSpotTrialPool(object):
    def __init__(self, **kwargs):
        pass

class BackgroundLocalWarmSpotPool(object):
    def __init__(self, **kwargs):
        pass

class Sensitivity(object):
    def __init__(self, **kwargs):
        pass
