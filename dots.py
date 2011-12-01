import time
import random
import pygame

from riglib import experiment, gendots, reward, options

class Dots(experiment.Pygame.Pygame):
    def __init__(self, *args, **kwargs):
        super(Dots, self).__init__(*args, **kwargs)
        self.width, self.height = self.surf.get_size()
        mask = gendots.squaremask()
        mid = self.height / 2 - mask.shape[0] / 2
        lc = self.width / 4 - mask.shape[1] / 2
        rc = 3*self.width / 4 - mask.shape[1] / 2

        self.mask = mask
        self.coords = (lc, mid), (rc, mid)

    def draw_frame(self):
        self.surf.blit(self.sleft, self.coords[0])
        self.surf.blit(self.sright, self.coords[1])
    
    def _start_reward(self):
        '''Dispense reward'''
        if reward is not None:
            reward.reward(reward_time*1000.)
    
    def _start_trial(self):
        left, right, flat = gendots.generate(self.mask)
        if random.random() > options['flat_proportion']:
            self._popout = True
            sleft = pygame.surfarray.make_surface((left>0).T)
            sright = pygame.surfarray.make_surface((right>0).T)
            sleft.set_palette([(0,0,0,255), (255,255,255,255)])
            sright.set_palette([(0,0,0,255), (255,255,255,255)])
            self.sleft, self.sright = sleft, sright
        else:
            self._popout = False
            sflat = pygame.surfarray.make_surface(flat>0)
            sflat.set_palette([(0,0,0,255), (255,255,255,255)])
            self.sleft, self.sright = sflat, sflat
    
    def _test_premature(self, ts):
        return self.event is not None
    
    def _test_correct(self, ts):
        return (not self._popout and ts > options['ignore_time']) or (self._popout and self.event is not None)

    def _test_incorrect(self, ts):
        return (self._popout and ts > options['ignore_time']) or (not self._popout and self.event is not None)