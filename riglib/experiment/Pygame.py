import os
import pygame

from __init__ import LogExperiment, traits

class Pygame(LogExperiment):
    background = (0,0,0)
    fps = 60

    timeout_time = traits.Float(4.)
    penalty_time = traits.Float(5.)
    reward_time = traits.Float(5.)

    def __init__(self, **kwargs):
        super(Pygame, self).__init__(**kwargs)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "2560,0"
        os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
        pygame.init()

        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        pygame.display.set_mode((3840,1080), flags)

        self.surf = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.event = None
    
    def draw_frame(self):
        raise NotImplementedError
    
    def clear_screen(self):
        self.surf.fill(self.background)
        pygame.display.flip()
    
    def _get_event(self):
        for e in pygame.event.get(pygame.KEYDOWN):
            return (e.key, e.type)
    
    def flip_wait(self):
        pygame.display.flip()
        self.event = self._get_event()
        self.clock.tick_busy_loop(self.fps)
    
    def _while_wait(self):
        self.surf.fill(self.background)
        self.flip_wait()
    
    def _while_trial(self):
        self.draw_frame()
        self.flip_wait()
    
    def _while_reward(self):
        self.clear_screen()
    def _while_penalty(self):
        self.clear_screen()
        
    def _test_start_trial(self, ts):
        return self.event is not None
    
    def _test_correct(self, ts):
        raise NotImplementedError
    
    def _test_incorrect(self, ts):
        raise NotImplementedError
    
    def _test_timeout(self, ts):
        return ts > self.timeout_time
    
    def _test_post_reward(self, ts):
        return ts > self.reward_time
    
    def _test_post_penalty(self, ts):
        return ts > self.penalty_time
