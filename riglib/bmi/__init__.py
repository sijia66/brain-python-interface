'''Needs docs'''


import numpy as np
import tables

from plexon import plexfile, psth
from riglib.nidaq import parse
import traceback

def load(decoder_fname):
    """Re-load decoder from pickled object"""
    decoder = pickle.load(open(decoder_fname, 'rb'))
    return decoder


class BMI(object):
    '''A BMI object, for filtering neural data into BMI output. Should be called decoder, not BMI.'''

    def __init__(self, cells, binlen=.1, **kwargs):
        '''All BMI objects must be pickleable objects with two main methods -- the init method trains
        the decoder given the inputs, and the call method actually does real time filtering'''
        self.files = kwargs
        self.binlen = binlen
        self.units = np.array(cells).astype(np.int32)

        self.psth = psth.SpikeBin(self.units, binlen)

    def __call__(self, data, task_data=None):
        psth = self.psth(data)
        if task_data is not None:
            task_data['bins'] = psth
        return psth

    def __setstate__(self, state):
        self.psth = psth.SpikeBin(state['cells'], state['binlen'])
        del state['cells']
        self.__dict__.update(state)

    def __getstate__(self):
        state = dict(cells=self.units)
        exclude = set(['plx', 'psth'])
        for k, v in self.__dict__.items():
            if k not in exclude:
                state[k] = v
        return state

    def get_data(self):
        raise NotImplementedError


import kfdecoder
import train

class AdaptiveBMI(object):
    def __init__(self, decoder, learner, updater):
        self.decoder = decoder
        self.learner = learner
        self.updater = updater
        self.param_hist = []

        self.clda_input_queue = self.updater.work_queue
        self.clda_output_queue = self.updater.result_queue
        self.updater.start()

    def __call__(self, spike_obs, target_pos, task_state, pos_inds=[0,1,2], *args, **kwargs):
        prev_state = self.decoder.get_state()
        update_flag=False
        # run the decoder
        #print kwargs
        self.decoder.predict(spike_obs, target=target_pos, **kwargs)
        decoded_state = self.decoder.get_state()
        
        # send data to learner
        if len(spike_obs) < 1:
            spike_counts = np.zeros((self.decoder.bin_spikes.nunits,))
        elif spike_obs.dtype == kfdecoder.python_plexnet_dtype:
            spike_counts = self.decoder.bin_spikes(spike_obs)
        else:
            spike_counts = spike_obs
        self.learner(spike_counts, prev_state[pos_inds], target_pos, task_state)

        try:
            new_params=None
            new_params = self.clda_output_queue.get_nowait()
        except:
            import os
            homedir = os.getenv('HOME')
            logfile = os.path.join(homedir, 'Desktop/log')
            f = open(logfile, 'w')
            traceback.print_exc(file=f)
            f.close()

        if new_params is not None:
            param_hist_data = list(new_params) + [self.intended_kin, self.spike_counts]
            self.param_hist.append(param_hist_data)
            self.decoder.update_params(new_params)
            self.learner.enable()
            update_flag = True
            print "updated params"

        if self.learner.is_full():
            self.intended_kin, self.spike_counts = self.learner.get_batch()
            rho = self.updater.rho
            #### TODO remove next line and make a user option instead
            drives_neurons = np.array([False, False, False, True, False, True, True])
            #drives_neurons = np.array([False, False, True, True, True])
            clda_data = (self.intended_kin, self.spike_counts, rho, self.decoder.kf.C, self.decoder.kf.Q, drives_neurons, self.decoder.mFR, self.decoder.sdFR)
            print self.intended_kin.shape
            print self.intended_kin[:,0]

            if 1:
                new_params = self.updater.calc(*clda_data)
                self.decoder.update_params(new_params)
            else:
                self.clda_input_queue.put(clda_data)
                self.learner.disable()

        return decoded_state, update_flag

    def __del__(self):
        self.updater.stop()
