import pygame
import sys
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from buttons import *
from classifier import event, folder_img
import re

def main():
    pygame.init()
    pygame.display.set_caption('Bound Shapes by YC')
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(size = (S_WIDTH+150,S_HEIGHT+50), flags = 0)
    surface = screen.convert()

    count = 1
    count_sub = count
    match = None
    todraw = []
    rad  = []
    redo, redorad = [], []
    label1, label2 = [], []

    pictures, txt = folder_img(dir, txtfolder)
    key_index = 0
    img_dir = None
    match = txt[0]
    index, prevind = 0, 0

    if len(pictures) >  0:
        img = plt.imread(f"{dir}/{pictures[0]}")
        img_dir = pygame.image.load(f"{dir}/{pictures[0]}").convert()
        height = img.shape[0]
        width = img.shape[1]
        print(f"This image has original height {height} and width {width}")
        img_dir = pygame.transform.scale(img_dir,(S_WIDTH,S_HEIGHT))
    if match != None:
       try:
            with open(f'{txtfolder}/{match}', 'r') as f:
               labels = f.read()
            
            labels = labels.strip().split('\n')
            print(labels)
            labels = np.array([list(map(int, label.split('\t'))) for label in labels])
            lines = np.argsort(labels[:, 2], axis=0)
            lines = labels[lines]
            lines = lines[::-1]
            print(f"THIS IS {lines}")         
            for l in lines:
               as_list = l
               todraw+= [(20+int(int(as_list[0]) *S_WIDTH/width),
                         int(S_HEIGHT/height*int(as_list[1]))+20)]
               rad+= [int(float(as_list[2])*S_WIDTH/width)]
               print(f"todraw is {todraw}")
               try:
                   label1.append(as_list[3])
               except:
                   label1.append(0)
               try:
                   label2.append(as_list[4])
               except:
                   label2.append(0)
       except:
            t = 0
    
 
    r = pygame.Rect((20, 20),(S_WIDTH, S_HEIGHT))
    r1 = pygame.Rect((S_HEIGHT,0),(170,S_WIDTH + 20))
    r2 = pygame.Rect((0,S_WIDTH),(S_HEIGHT , 170))
    r3 = pygame.Rect((0,0),(20, S_WIDTH+20))
    r4 = pygame.Rect((0,0),(S_HEIGHT, 20))
    r5 = pygame.Rect((250,0),(250, 20))

    myfont = pygame.font.SysFont("fontname", 20)
    pygame.draw.rect(surface, (255,0,0), r5)  
    pygame.draw.rect(surface, clr2, r1)
    pygame.draw.rect(surface, clr2, r2)
    pygame.draw.rect(surface, clr2, r3)
    pygame.draw.rect(surface, clr2, r4)
    text1 = myfont.render(f"{pictures[index]} saving to {txt[index]} ", 1, (255,255,255))
    surface.blit(text1, (250 ,0))
    screen.blit(surface, (0,0))
    if img_dir != None:
         screen.blit(img_dir,(20,20))
    else: 
         print("no image")
         pygame.draw.rect(surface, clr2, r)
    for i in range(0,len(todraw)):
        pygame.draw.circle(screen, clr3, (todraw[0]), int(rad[0]),bor_size)
        pygame.draw.circle(screen, clr1, (todraw[i]), int(rad[i]),bor_size)
    draw_buttons(screen,8)
    print(label1, label2)
    text_render(screen, screen, label1, label2, todraw)
   # pygame.display.update()
    pos = None
    myfont = pygame.font.SysFont("fontname", 50)
    while True:
         clock.tick(90)
         draw_buttons(surface,8)
        # text_render(surface, surface, label1, label2, todraw)
         pos, todraw, rad, redo, redorad, index, prevind, txt, count, key_index,  label1, label2 = \
	event(surface, pos, todraw,rad, screen, redo,redorad, index,pictures, prevind, txt, height, width, count, key_index, label1, label2, count_sub)
        # text_render(surface, surface, label1, label2, todraw)
         r5 = pygame.Rect((250,0),(250, 20))
         pygame.draw.rect(surface, (255,0,0), r5)
         #screen.blit(surface, (0,0))
         myfont = pygame.font.SysFont("fontname", 20)
         text1 = myfont.render(f"{pictures[index]} saving to {txt[index]} ", 1, (255,255,255))
         surface.blit(text1, (250 ,0))
         pygame.display.update()
        # for i in range(0,len(todraw)):
         #      pygame.draw.circle(screen, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
          #     pygame.draw.circle(screen, clr1, (todraw[i]), int(rad[i]),bor_size)
         pygame.display.update()
#         text1 = myfont.render("L1", 1, (96, 108, 118))
         if img_dir != None:
             img_dir = pygame.image.load(f"{dir}/{pictures[index]}")
             img_dir = pygame.transform.scale(img_dir,(S_WIDTH,S_HEIGHT))
             surface.blit(img_dir,(20,20))
             if index != prevind:
                 print(f"count is  {count}")
                 print(f"This is Picture {index}")
                 todraw, rad, redo, redorad = [],[],[],[]
                 match = txt[index]
                 key_index = 0
                 label1, label2 = [], []
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
                       #info = open(f'{txtfolder}/{match}', 'r')
                       #with info as draw:
                       #   lines = draw.readlines()
                       with open(f'{txtfolder}/{match}', 'r') as f:
                          labels = f.read()
            
                       labels = labels.strip().split('\n')
                       labels = np.array([list(map(int, label.split('\t'))) for label in labels])
                       print(f"This is {labels}")        
                       lines = np.argsort(labels[:, -2], axis=0)
                       print(lines)
                       lines = labels[lines]
                       lines = lines[::-1]
                       for l in lines:
                          #as_list = re.split("\t|\n", l)
                          as_list = l
                          todraw+= [(  20+  int( int(as_list[0]) *S_WIDTH/width),
                                   int(S_HEIGHT/height*int(as_list[1])) +20   )]
                          rad+= [int(float(as_list[2])*S_WIDTH/width)]
                          try:
                                 label1.append(as_list[3])
                          except:
                                 label1.append(0)
                          try:
                                 label2.append(as_list[4])
                          except:
                                 label2.append(0)
                    except:
                        print('EXCEPT in MAIN.PY')
                        t = 0
                 draw_buttons(surface,100)
                 text_render(surface, screen, label1, label2, todraw)
                 for i in range(0,len(todraw)):
                     pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                     pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                 screen.blit(surface, (0,0))
                 pygame.display.update()
                 prevind = index
         else:
             pygame.draw.rect(surface, clr2, r)
    return


if __name__ == "__main__":
    main()
