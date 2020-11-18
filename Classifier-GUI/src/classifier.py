import pygame
import sys
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from buttons import*

def event(surface, pos, todraw,rad, screen, redo, redorad, index, pictures, prevind, txt, height, width, save_index, key_index,  label1, label2, count_sub):
    r1 = pygame.Rect((S_HEIGHT+20,0),(170, S_WIDTH+150))
    r2 = pygame.Rect((0,S_WIDTH+20),(S_HEIGHT+200, 170))
    r3 = pygame.Rect((0,0),(20, S_WIDTH+200))
    r4 = pygame.Rect((0,0),(S_HEIGHT+20, 20))
    key_state = pygame.key.get_pressed()
    for event in pygame.event.get():
       if event.type == pygame.QUIT:
           pygame.quit()
           sys.exit()
       elif event.type == pygame.KEYDOWN:
           keys = pygame.key.get_pressed()     
           if (event.key == pygame.K_k): # or (keys[pygame.K_r]):
                 key_index = (key_index+1)%len(todraw)   
                 text_render(surface, surface,label1, label2, todraw)            
                 for i in range(0,len(todraw)):
                      pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                      pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                 screen.blit(surface, (0,0))
                 pygame.display.update()
           elif event.key == pygame.K_g:
                for i in range(0,len(todraw)):
                          pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                          pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                screen.blit(surface, (0,0))
                pygame.display.update()
           elif (event.key == pygame.K_RIGHT)or (keys[pygame.K_RIGHT]):
               prevind = index
               save_index = (index+1)%len(pictures)+ count_sub
               index = (index+ 1)%len(pictures)
               text_render(surface, surface,label1, label2, todraw)
           elif (event.key == pygame.K_LEFT)or (keys[pygame.K_LEFT]):
               prevind = index
               save_index = (index-1)%len(pictures)+ count_sub
               index = (index-1)%len(pictures)
               text_render(surface, surface,label1, label2, todraw)
           elif (event.key == pygame.K_SPACE):
         #      for i in range(0,len(todraw)):
         #            pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
         #            pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
         #            screen.blit(surface, (0,0))
               img = pygame.surfarray.pixels3d(screen)
               img = img[20:S_WIDTH+20,20:S_HEIGHT+20,:]
               img = ndimage.rotate(img,90)
               img = img[::-1,:]
               if len(pictures)>0:
                   org_image = plt.imread(f"{dir}/{pictures[index]}")
            #plt.imsave(f"{dir_save}/{save_index}.jpeg", org_image)
                   with open(f"{txtsave}/{txt[index]}", 'w') as f:
                       for i in range(len(todraw)):
                           f.write(f"{int((todraw[i][0]-19)*width/S_WIDTH)}\t{int((todraw[i][1]-19)*height/S_HEIGHT)}\t{int(0.5+rad[i]*width/S_WIDTH)}\t{label1[i]}\t{label2[i]}\n")
               save_index = (index+1)%len(pictures) + count_sub
               prevind = index
               index = (index+ 1)%len(pictures)
           elif (event.key == pygame.K_c):
               if len(todraw)>0:
                   redo= todraw
                   todraw = []
                   redorad = rad
                   rad = []
               screen.blit(surface, (0,0))
               pygame.display.update()


           # Classify according to the keys
           elif (event.key == pygame.K_0):
                label1[key_index] = 0
                text_render(surface, surface,label1, label2, todraw)
                for i in range(0,len(todraw)):
                        pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                        pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                screen.blit(surface, (0,0))
                pygame.display.update()
           elif (event.key == pygame.K_1):
              label1[key_index] = 1
              text_render(surface, surface,label1, label2, todraw)
              for i in range(0,len(todraw)):
                      pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                      pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
              screen.blit(surface, (0,0))
              pygame.display.update()
           elif (event.key == pygame.K_2):
               label1[key_index] = 2
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_3):
               label1[key_index] = 3
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_4):
               label1[key_index] = 4
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_5):
               label1[key_index] = 5
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_6):
               label1[key_index] = 6
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_h):
               label2[key_index] = 72
               text_render(surface, surface,label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_t):
               label2[key_index] = 84
               text_render(surface, surface, label1, label2, todraw)
               for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
               pygame.display.update()
       elif event.type == pygame.KEYUP:
           if (event.key == pygame.K_g):
               text_render(surface, surface, label1, label2, todraw)
               for i in range(0,len(todraw)):
                    pygame.draw.circle(surface, clr3, (todraw[key_index]), int(rad[key_index]),bor_size)
                    pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update() 
    return pos, todraw, rad, redo, redorad, index, prevind, txt, save_index, key_index, label1, label2


def folder_img(dir, txt):
   pictures = sorted(os.listdir(dir))
   txtfiles = [] #sorted(os.listdir(txtfolder))  
   pic = []
   txt2 = []
   txt3 = []
   # test = [elem[:-3] for elem in pictures]
   test = [elem[:-3] for elem in txtfiles]
   for picture in pictures:
       if picture[-3:] == "jpg" or picture[-3:] == "png" or picture[-3:] == "peg":
           pic.append(picture)
   test_jpg = [elem[:-3] for elem in pic]
   test_jpeg = [elem[:-4] for elem in pic]
   for pic2 in pic:
       if (pic2[:-4] not in test) and (pic2[-3:] == "jpg" or pic2[-3:] == "png"):
           txtfiles.append(f"{pic2[:-4]}.txt")
       elif pic2[:-5] not in test:
           txtfiles.append(f"{pic2[:-5]}.txt")
   txtfiles = sorted(txtfiles)
   for txtf in txtfiles:
       if txtf[-4:] == ".csv" or txtf[-4:] == ".txt":
           txt2.append(txtf)
  # print(txt2)
   for t in txt2:
       if t[:-3] in test_jpg or t[:-3] in test_jpeg:
           txt3.append(t)
   txt3 = sorted(txt3)
   print(f" The images you have in the directory are {pic}")
   print(f" The txt files you have in the directory are {txt3}")
   return pic, txt3
