'''Needs docs'''

import numpy as np
import traceback
import re

def load(decoder_fname):
    """Re-load decoder from pickled object"""
    decoder = pickle.load(open(decoder_fname, 'rb'))
    return decoder

from bmi import BMI, AdaptiveBMI
import kfdecoder
import train
