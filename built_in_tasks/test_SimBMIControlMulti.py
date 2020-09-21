from bmimultitasks import SimBMIControlMulti, SimBMIControlMultiFAEnc
from features import SaveHDF
from features.simulation_features import BMISimExpTuner
import numpy as np
from riglib import experiment
import time

def get_enc_setup(sim_mode = 'toy'):
    # sim_mode:str 
    #   std:  mn 20 neurons
    #   'toy' # mn 4 neurons

    if sim_mode == 'toy':
        #by toy, w mn 4 neurons:
            #first 2 ctrl x velo
            #lst 2 ctrl y vel
        # build a observer matrix
        N_NEURONS = 4
        N_STATES = 7  # 3 positions and 3 velocities and an offset
        # build the observation matrix
        sim_C = np.zeros((N_NEURONS, N_STATES))
        # control x positive directions
        sim_C[0, :] = np.array([0, 0, 0, 1, 0, 0, 0])
        sim_C[1, :] = np.array([0, 0, 0, -1, 0, 0, 0])
        # control z positive directions
        sim_C[2, :] = np.array([0, 0, 0, 0, 0, 1, 0])
        sim_C[3, :] = np.array([0, 0, 0, 0, 0, -1, 0])
        

    elif sim_mode ==  'std':
        # build a observer matrix
        N_NEURONS = 20
        N_STATES = 7  # 3 positions and 3 velocities and an offset
        # build the observation matrix
        sim_C = np.zeros((N_NEURONS, N_STATES))
        # control x positive directions
        sim_C[0, :] = np.array([0, 0, 0, 1, 0, 0, 0])
        sim_C[1, :] = np.array([0, 0, 0, -1, 0, 0, 0])
        # control z positive directions
        sim_C[2, :] = np.array([0, 0, 0, 0, 0, 1, 0])
        sim_C[3, :] = np.array([0, 0, 0, 0, 0, -1, 0])
    else:
        raise Exception(f'not recognized mode {sim_mode}')
    
    return (N_NEURONS, N_STATES, sim_C)
    

# build a sequence generator
if __name__ == "__main__":

    #generate task params
    N_TARGETS = 8
    N_TRIALS = 4
    seq = SimBMIControlMulti.sim_target_seq_generator_multi(
        N_TARGETS, N_TRIALS)

    #neuron set up : 'std (20 neurons)' or 'toy (4 neurons)' 
    N_NEURONS, N_STATES, sim_C = get_enc_setup(sim_mode = 'toy')

    #sav everthing in a kw
    kwargs = dict()
    # set up assist level
    assist_level = (0.1, 0.1)
    kwargs['sim_C'] = sim_C
    kwargs['assist_level'] = assist_level

    #base_class = SimBMIControlMulti
    base_class = SimBMIControlMultiFAEnc
    #feats = [SaveHDF, BMISimExpTuner]
    #feats = [SaveHDF]
    feats = [BMISimExpTuner]
    
    Exp = experiment.make(base_class, feats=feats)
    print(Exp)

    exp = Exp(seq, **kwargs)
    exp.init()
    exp.run()  # start the task

