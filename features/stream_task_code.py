class TaskCodeStreamer(object):

    def set_state(self, condition, **kwargs):
        '''
        Extension of riglib.experiment.Experiment.set_state. Send the name of the next state to 
        plexon system and then proceed to the upstream set_state tasks.

        Parameters
        ----------
        condition : string
            Name of new state.
        **kwargs : dict 
            Passed to 'super' set_state function

        Returns
        -------
        None
        '''
        print(f'current state {condition}')
        super().set_state(condition, **kwargs)