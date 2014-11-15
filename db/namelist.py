'''
Lookup table for features, generators and tasks for experiments
'''

import numpy as np
from riglib import experiment, calibrations, bmi
from riglib.stereo_opengl.window import MatplotlibWindow

features = dict(
    autostart=experiment.features.Autostart, 
    adaptive_generator=experiment.features.AdaptiveGenerator,
    button=experiment.features.Button, 
    ignore_correctness=experiment.features.IgnoreCorrectness,
    reward_system = experiment.features.RewardSystem,
    eye_data=experiment.features.EyeData,
    joystick=experiment.features.Joystick,
    dual_joystick=experiment.features.DualJoystick,
    calibrated_eye=experiment.features.CalibratedEyeData,
    eye_simulate=experiment.features.SimulatedEyeData,
    fixation_start=experiment.features.FixationStart,
    motion_data=experiment.features.MotionData,
    motion_simulate=experiment.features.MotionSimulate, 
    motion_autoalign=experiment.features.MotionAutoAlign,
    bmi=experiment.features.PlexonBMI,
    blackrockbmi=experiment.features.BlackrockBMI,
    saveHDF=experiment.features.SaveHDF,
    relay_plexon=experiment.features.RelayPlexon,
    relay_plexbyte=experiment.features.RelayPlexByte,
    relay_blackrockbyte=experiment.features.RelayBlackrockByte,
    norm_firingrates=experiment.features.NormFiringRates,
    ttl_reward=experiment.features.TTLReward,
    juice_log=experiment.features.JuiceLogging,
    single_video=experiment.features.SingleChannelVideo,
    exp_display=MatplotlibWindow,
)

## Build the list of generators
try:
    from tasklist import tasks
except ImportError:
    print 'create a file/module tasklists.py and create the task dictionaries inside it!'
    tasks = dict()

from itertools import izip

# Derive generator functions from the tasklist (all generatorfunctions should be staticmethods of a task)
generator_names = []
generator_functions = []
for task in tasks:
    task_cls = tasks[task]
    generator_function_names = task_cls.sequence_generators
    gen_fns = [getattr(task_cls, x) for x in generator_function_names]
    for fn_name, fn in izip(generator_function_names, gen_fns):
        if fn in generator_functions:
            pass
        else:
            generator_names.append(fn_name)
            generator_functions.append(fn)

generators = dict()
for fn_name, fn in izip(generator_names, generator_functions):
    generators[fn_name] = fn


from tracker import models
class SubclassDict(dict):
    '''
    A special dict that returns the associated Django database model 
    if the queried item is a subclass of any of the keys
    '''
    def __getitem__(self, name):
        try:
            return super(self.__class__, self).__getitem__(name)
        except KeyError:
            for inst, model in self.items():
                if issubclass(name, inst):
                    return model
        raise KeyError
        
instance_to_model = SubclassDict( {
    calibrations.Profile:models.Calibration,
    calibrations.AutoAlign:models.AutoAlignment,
    bmi.BMI: models.Decoder,
    bmi.Decoder: models.Decoder,
} )


bmi_algorithms = dict(
    KFDecoder=bmi.train.train_KFDecoder,
    PPFDecoder=bmi.train.train_PPFDecoder,
)

bmi_training_pos_vars = ['cursor', 'joint_angles']

bmi_state_space_models=dict(
    Endpt2D=bmi.train.endpt_2D_state_space,
    Endpt3D=bmi.train.endpt_3D_state_space,
    Tentacle=bmi.train.tentacle_2D_state_space,
    Armassist=bmi.train.armassist_state_space,
    Rehand=bmi.train.rehand_state_space,
    ISMORE=bmi.train.ismore_state_space,
)

extractors = dict(
    spikecounts = bmi.extractor.BinnedSpikeCountsExtractor,
    LFPpowerMTM = bmi.extractor.LFPMTMPowerExtractor,
    LFPpowerBPF = bmi.extractor.LFPButterBPFPowerExtractor,
    EMGAmplitude = bmi.extractor.EMGAmplitudeExtractor,
)

default_extractor = "spikecounts"

bmi_update_rates = [10, 20, 30, 60, 120, 180]
