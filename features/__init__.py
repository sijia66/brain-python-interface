'''
Module for the core "features" that can be used to extend and customize a 
task/experiment by multiple inheritance.
'''
from .hdf_features import SaveHDF


built_in_features = dict(
    saveHDF=SaveHDF,
)

# >>> features.built_in_features['autostart'].__module__
# 'features.generator_features'
# >>> features.built_in_features['autostart'].__qualname__
# 'Autostart'