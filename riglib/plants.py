#!/usr/bin/python
'''
Representations of plants (control systems)
'''
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np

import sys
import time
import socket
import select
import numpy as np
from collections import namedtuple

from riglib import source

import struct
import math



class Plant(object):
    '''
    Generic interface for task-plant interaction
    '''
    hdf_attrs = []
    def __init__(self, *args, **kwargs):
        pass
        
    def drive(self, decoder):
        '''
        Call this function to 'drive' the plant to the state specified by the decoder

        Parameters
        ----------
        decoder : bmi.Decoder instance 
            Decoder used to estimate the state of/control the plant 

        Returns
        -------
        None
        '''
        # Instruct the plant to go to the decoder-specified intrinsic coordinates
        # decoder['q'] is a special __getitem__ case. See riglib.bmi.Decoder.__getitem__/__setitem__
        self.set_intrinsic_coordinates(decoder['q'])

        # Not all intrinsic coordinates will be achievable. So determine where the plant actually went
        intrinsic_coords = self.get_intrinsic_coordinates()

        # Update the decoder state with the current state of the plant, after the last command
        if not np.any(np.isnan(intrinsic_coords)):
            decoder['q'] = self.get_intrinsic_coordinates()

    def get_data_to_save(self):
        '''
        Get data to save regarding the state of the plant on every iteration of the event loop

        Parameters
        ----------
        None

        Returns
        -------
        dict: 
            keys are strings, values are np.ndarray objects of data values
        '''
        return dict()

    def init(self):
        '''
        Secondary initialization after object construction. Does nothing by default
        '''
        pass

    def start(self):
        '''
        Start any auxiliary processes used by the plant
        '''
        pass

    def stop(self):
        '''
        Stop any auxiliary processes used by the plant
        '''        
        pass

    def init_decoder(self, decoder):
        decoder['q'] = self.get_intrinsic_coordinates()


###################################################
##### Virtual plants for specific experiments #####
###################################################
class CursorPlant(Plant):
    '''
    Create a plant which is a 2-D or 3-D cursor on a screen/stereo display
    '''
    hdf_attrs = [('cursor', 'f8', (3,))]
    def __init__(self, endpt_bounds=None, cursor_radius=0.4, cursor_color=(.5, 0, .5, 1), starting_pos=np.array([0., 0., 0.]), vel_wall=True, **kwargs):
        self.endpt_bounds = endpt_bounds
        self.position = starting_pos
        self.starting_pos = starting_pos
        self.cursor_radius = cursor_radius
        self.cursor_color = cursor_color
        self._pickle_init()
        self.vel_wall = vel_wall

    def _pickle_init(self):
        self.cursor = Sphere(radius=self.cursor_radius, color=self.cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def draw(self):
        self.cursor.translate(*self.position, reset=True)

    def get_endpoint_pos(self):
        return self.position

    def set_endpoint_pos(self, pt, **kwargs):
        self.position = pt
        self.draw()

    def get_intrinsic_coordinates(self):
        return self.position

    def set_intrinsic_coordinates(self, pt):
        self.position = pt
        self.draw()

    def set_visibility(self, visible):
        self.visible = visible
        if visible:
            self.graphics_models[0].attach()
        else:
            self.graphics_models[0].detach()

    def _bound(self, pos, vel):
        pos = pos.copy()
        vel = vel.copy()
        if self.endpt_bounds is not None:
            if pos[0] < self.endpt_bounds[0]: 
                pos[0] = self.endpt_bounds[0]
                if self.vel_wall: vel[0] = 0
            if pos[0] > self.endpt_bounds[1]: 
                pos[0] = self.endpt_bounds[1]
                if self.vel_wall: vel[0] = 0

            if pos[1] < self.endpt_bounds[2]: 
                pos[1] = self.endpt_bounds[2]
                if self.vel_wall: vel[1] = 0
            if pos[1] > self.endpt_bounds[3]: 
                pos[1] = self.endpt_bounds[3]
                if self.vel_wall: vel[1] = 0

            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                if self.vel_wall: vel[2] = 0
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
                if self.vel_wall: vel[2] = 0
        return pos, vel

    def drive(self, decoder):
        pos = decoder['q'].copy()
        vel = decoder['qdot'].copy()

        pos, vel = self._bound(pos, vel)
        
        decoder['q'] = pos
        decoder['qdot'] = vel
        super(CursorPlant, self).drive(decoder)

    def get_data_to_save(self):
        return dict(cursor=self.position)



class onedimLFP_CursorPlant(CursorPlant):
    '''
    A square cursor confined to vertical movement
    '''
    hdf_attrs = [('lfp_cursor', 'f8', (3,))]

    def __init__(self, endpt_bounds, *args, **kwargs):
        self.lfp_cursor_rad = kwargs['lfp_cursor_rad']
        self.lfp_cursor_color = kwargs['lfp_cursor_color']
        args=[(), kwargs['lfp_cursor_color']]
        super(onedimLFP_CursorPlant, self).__init__(endpt_bounds, *args, **kwargs)


    def _pickle_init(self):
        self.cursor = Cube(side_len=self.lfp_cursor_rad, color=self.lfp_cursor_color)
        self.cursor.translate(*self.position, reset=True)
        self.graphics_models = [self.cursor]

    def drive(self, decoder):
        pos = decoder.filt.get_mean()
        pos = [-8, -2.2, pos]

        if self.endpt_bounds is not None:
            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
               
            self.position = pos
            self.draw()

    def turn_off(self):
        self.cursor.detach()

    def turn_on(self):
        self.cursor.attach()

    def get_data_to_save(self):
        return dict(lfp_cursor=self.position)


class twodimLFP_CursorPlant(onedimLFP_CursorPlant):
    '''Same as 1d cursor but assumes decoder returns array '''
    def drive(self, decoder):
        #Pos = (Left-Right, 0, Up-Down)
        pos = decoder.filt.get_mean()
        pos = [pos[0], -2.2, pos[2]]
        #pos = [-8, -2.2, pos[2]]

        if self.endpt_bounds is not None:
            if pos[2] < self.endpt_bounds[4]: 
                pos[2] = self.endpt_bounds[4]
                
            if pos[2] > self.endpt_bounds[5]: 
                pos[2] = self.endpt_bounds[5]
               
            self.position = pos
            self.draw()


