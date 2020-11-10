import pygame
import sys
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from buttons import*
from boundbox import event, folder_img
import re

def main():
    pygame.init()
    pygame.display.set_caption('Bound Shapes by YC')
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size = (S_WIDTH+150,S_HEIGHT+50), flags = 0)
    surface = screen.convert()
    match = None
    todraw = []
    rad = []
    redo, redorad = [], []
    pictures, txt = folder_img(dir, txtfolder)
   
    print(pictures[0])
    img_dir = None
    match = txt[0]
    index, prevind = 0, 0
    if len(pictures) > 0:
        img = plt.imread(f"{dir}/{pictures[0]}")
        img_dir = pygame.image.load(f"{dir}/{pictures[0]}").convert()
        height = img.shape[0]
        width = img.shape[1]
        print(f"This image has original height {height} and width {width}")
        img_dir = pygame.transform.scale(img_dir,(S_WIDTH,S_HEIGHT))

    if match != None:
       try:
#            print(match)
            #info = open(f'{txtfolder}/{match}', 'r')
            '''
            with info as draw:
               lines = draw.readlines()
            
            labels = [i.strip() for i in lines]
            print(lines)
            print(labels)
            
            #lines.sort(key=lambda x[2]: , reverse = True)
            '''
            with open(f'{txtfolder}/{match}', 'r') as f:
               labels = f.read()
            
            labels = labels.strip().split('\n')
            labels = np.array([list(map(int, label.split('\t'))) for label in labels])

            lines = np.argsort(labels[:, 2], axis=0)
            lines = labels[lines]
            print(lines)
            
            for l in lines:
               
               #as_list = re.split("\t|\n", l)
               as_list = l
               
                
               todraw+= [(  20+  int( int(as_list[0]) *S_WIDTH/width),
                         int(S_HEIGHT/height*int(as_list[1])) +20   )]
             #  print(f"todraw is {len(todraw)}") 
               rad+= [int(float(as_list[2])*S_WIDTH/width)]
             #  print(rad)
       except:
            t = 0
    
 
    r = pygame.Rect((20, 20),(S_WIDTH, S_HEIGHT))
    r1 = pygame.Rect((S_HEIGHT,0),(170,S_WIDTH + 150))
    r2 = pygame.Rect((0,S_WIDTH),(S_HEIGHT , 170))
    r3 = pygame.Rect((0,0),(20, S_WIDTH+170))
    r4 = pygame.Rect((0,0),(S_HEIGHT, 20))

    pygame.draw.rect(surface, clr2, r1)
    pygame.draw.rect(surface, clr2, r2)
    pygame.draw.rect(surface, clr2, r3)
    pygame.draw.rect(surface, clr2, r4)
    screen.blit(surface, (0,0))
    if img_dir != None:
         screen.blit(img_dir,(20,20))
    else: 
         print("no image")
         pygame.draw.rect(surface, clr2, r)
    for i in range(0,len(todraw)):
        pygame.draw.circle(screen, clr1, (todraw[i]), int(rad[i]),bor_size)
    draw_buttons(screen,8)
    text_render(surface, screen)
    pygame.display.update()
    pos = None
    myfont = pygame.font.SysFont("fontname", 50)
    count = 410
    while True:
         clock.tick(90)
         draw_buttons(surface,8)
         text_render(surface, surface)
         pos, todraw, rad,redo,redorad,index, prevind,txt,count = event(surface, pos, todraw,rad, screen, redo,redorad, index,pictures, prevind, txt, height, width,count)
         
         text1 = myfont.render("L1", 1, (96, 108, 118))
         if img_dir != None:
             img_dir = pygame.image.load(f"{dir}/{pictures[index]}")
             img_dir = pygame.transform.scale(img_dir,(S_WIDTH,S_HEIGHT))
             surface.blit(img_dir,(20,20))
             if index != prevind:
                 print(f"This is Picture {index}")
                 todraw, rad, redo, redorad = [],[],[],[]
                 match = txt[index]
                 print(match)
                 if len(pictures) >  0:
                    img = plt.imread(f"{dir}/{pictures[index]}")
                    img_dir = pygame.image.load(f"{dir}/{pictures[index]}").convert()
                    height = img.shape[0]
                    width = img.shape[1]
                    print(f"This image has original height {height} and width {width}")
                    img_dir = pygame.transform.scale(img_dir,(S_WIDTH,S_HEIGHT))
                 if match != None:
                    try:
                       info = open(f'{txtfolder}/{match}', 'r')
                       with info as draw:
                          lines = draw.readlines()
                       for l in lines:
                          as_list = re.split("\t|\n", l)
                          todraw+= [(  20+  int( int(as_list[0]) *S_WIDTH/width),
                                   int(S_HEIGHT/height*int(as_list[1])) +20   )]
                          rad+= [int(float(as_list[2])*S_WIDTH/width)]
                    except:
                       t = 0
                 draw_buttons(surface,100)
                 text_render(surface, surface)
                 for i in range(0,len(todraw)):
                     pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                 screen.blit(surface, (0,0))
                 pygame.display.update()
                 prevind = index
         else:
             pygame.draw.rect(surface, clr2, r)
    return


if __name__ == "__main__":
    main()
