from bmimultitasks import SimBMIControlMulti, SimBMIControlMultiFAEnc
from test_SimBMIControlMulti import get_enc_setup

from features import SaveHDF
from riglib import experiment

import time
import numpy as np


"""
this task uses 
"""


# build a sequence generator
if __name__ == "__main__":

    #generate task params
    N_TARGETS = 8
    N_TRIALS = 6
    seq = SimBMIControlMulti.sim_target_seq_generator_multi(
        N_TARGETS, N_TRIALS)

    #neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
    N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'toy')

    # set up assist level
    #assist_level = (0.1, 0.1)

    #base_class = SimBMIControlMulti
    base_class = SimBMIControlMulti
    feats = [SaveHDF]

    #sav everthing in a kw
    kwargs = dict()
    kwargs['sim_C'] = sim_C
    #kwargs['assist_level'] = assist_level
    Exp = experiment.make(base_class, feats=feats)
    print(Exp)

    exp = Exp(seq, **kwargs)
    exp.init()
    exp.run()  # start the task

