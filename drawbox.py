import os
import sys
import pygame

os.environ['SDL_VIDEO_WINDOW_POS'] = "1024,0"
os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"
pygame.init()

flags = pygame.DOUBLEBUF | pygame.HWSURFACE
pygame.display.set_mode((3840,1080), flags)

surf = pygame.display.get_surface()
coords = map(int, sys.argv[1:])
c1 = coords[:]
c1[2] = coords[2] - coords[0]
c1[3] = coords[3] - coords[1]
c2 = c1[:]
c2[0] += 1920

while True:
	surf.fill((0,0,0))
	surf.fill((255,255,255), rect=c1)
	surf.fill((255,255,255), rect=c2)
	pygame.display.flip()


