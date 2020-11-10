import pygame
import sys
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from buttons import*

def event(surface, pos, todraw,rad, screen, redo, redorad,  index,pictures, prevind, txt, height, width,save_index):
    
    r1 = pygame.Rect((S_HEIGHT+20,0),(170, S_WIDTH+150))
    r2 = pygame.Rect((0,S_WIDTH+20),(S_HEIGHT+20, 170))
    r3 = pygame.Rect((0,0),(20, S_WIDTH+200))
    r4 = pygame.Rect((0,0),(S_HEIGHT+200, 20)) 
    for event in pygame.event.get():
       if event.type == pygame.QUIT:
           pygame.quit()
           sys.exit()
       elif event.type == pygame.KEYDOWN:
           keys = pygame.key.get_pressed() 
           if (event.key == pygame.K_RETURN):
               for i in range(0,len(todraw)):
                     pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                     screen.blit(surface, (0,0))
               img = pygame.surfarray.pixels3d(screen)
               img = img[20:S_WIDTH+20,20:S_HEIGHT+20,:]
               img = ndimage.rotate(img,90)
               img = img[::-1,:]
               if len(pictures)>0:
                   org_image = plt.imread(f"bad_images/{pictures[index]}")
                   plt.imsave(f"{dir_save}/{save_index}.jpeg", org_image)
                   with open(f"{txtsave}/{save_index}.txt", 'w') as f:
                       for i in range(len(todraw)):
                           f.write(f"{int((todraw[i][0]-19)*width/S_WIDTH)}\t{int((todraw[i][1]-19)*height/S_HEIGHT)}\t{int(0.5+rad[i]*width/S_WIDTH)}\n")
                   save_index += 1 
           elif (event.key == pygame.K_d):
               if len(todraw)>0:
                   redo+= [todraw[-1]]
                   
                   todraw = todraw[:-1]
                   redorad+= [rad[-1]]
                   rad = rad[:-1]
               for i in range(0,len(todraw)):
                    pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_r) or (keys[pygame.K_r]):
               if len(redo) >0:
                   todraw+=[redo[-1]]
                   rad+=[redorad[-1]]
                   redo = redo[:-1]
                   redorad = redorad[:-1]
               for i in range(0,len(todraw)):
                    pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
               screen.blit(surface, (0,0))
               pygame.display.update()
           elif (event.key == pygame.K_RIGHT)or (keys[pygame.K_RIGHT]):
               prevind = index
               index = (index+ 1)%len(pictures)          
           elif (event.key == pygame.K_LEFT)or (keys[pygame.K_LEFT]):
               prevind = index
               index = (index-1)%len(pictures)
           elif (event.key == pygame.K_SPACE):
               for i in range(0,len(todraw)):
                     pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                     screen.blit(surface, (0,0))
               img = pygame.surfarray.pixels3d(screen)
               img = img[20:S_WIDTH+20,20:S_HEIGHT+20,:]
               img = ndimage.rotate(img,90)
               img = img[::-1,:]
               if len(pictures)>0:
                   org_image = plt.imread(f"bad_images/{pictures[index]}")
                   plt.imsave(f"{dir_save}/{save_index}.jpeg", org_image)
                   with open(f"{txtsave}/{save_index}.txt", 'w') as f:
                       for i in range(len(todraw)):
                           f.write(f"{int((todraw[i][0]-19)*width/S_WIDTH)}\t{int((todraw[i][1]-19)*height/S_HEIGHT)}\t{int(0.5+rad[i]*width/S_WIDTH)}\n")
                   save_index += 1 
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



       elif event.type == pygame.MOUSEBUTTONDOWN:
           pos2 = event.pos
           if pos != None and  pos2[0]<(S_WIDTH+20) and pos2[1]<S_HEIGHT+20:
               pos2 = event.pos
               radius = sqrt((pos[0] - pos2[0])**2+ (pos[1] - pos2[1])**2)//2
               todraw += [(int((pos[0]+pos2[0])/2), int((pos[1]+pos2[1])/2))]
               rad += [radius]
               pos = None
               break
           if pos2[0]<(S_WIDTH+20) and pos2[1]<S_HEIGHT+20:
               pos = event.pos
           else:
               pos3 = event.pos
               if (pos3[0] >  S_WIDTH +58 and pos3[0] <  S_WIDTH +118) and (pos3[1] > 102 and pos3[1] < 152):
                   for i in range(0,len(todraw)):
                        pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                        screen.blit(surface, (0,0))
                   img = pygame.surfarray.pixels3d(screen)
                   img = img[20:S_WIDTH+20,20:S_HEIGHT+20,:]
                   img = ndimage.rotate(img,90)
                   img = img[::-1,:]
                   draw_buttons(surface,1)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   if len(pictures)>0:
                       plt.imsave(f"{dir_save}/{pictures[index]}", img)
                       with open(f"{txtsave}/{txt[index]}", 'w') as f:
                           for i in range(len(todraw)):
                               f.write(f"{int((todraw[i][0]-19)*width/S_WIDTH)}\t{int((todraw[i][1]-19)*height/S_HEIGHT)}\t{int(0.5+rad[i]*width/S_WIDTH)}\n")
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
               elif (pos3[0] >  S_WIDTH +58 and pos3[0] <  S_WIDTH +118) and (pos3[1] > 202 and pos3[1] < 252):
                   if len(todraw)>0:
                       redo+= [todraw[-1]]
                       todraw = todraw[:-1]
                       redorad+= [rad[-1]]
                       rad = rad[:-1]
                   for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                   draw_buttons(surface,3)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
               elif (pos3[0] > S_WIDTH + 68 and pos3[0] < S_WIDTH + 118) and (pos3[1] > 302 and pos3[1] < 352):
                   if len(redo) >0:
                       todraw+=[redo[-1]]
                       rad+=[redorad[-1]]
                       redo = redo[:-1]
                       redorad = redorad[:-1]
                   for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                   draw_buttons(surface,4)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
               elif (pos3[0] > S_WIDTH + 58 and pos3[0] < S_WIDTH + 128) and (pos3[1] > 402 and pos3[1] < 427):
                   prevind = index
                   index = (index+ 1)%len(pictures)
                   for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                   draw_buttons(surface,5)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
               elif (pos3[0] > S_WIDTH + 58 and pos3[0] < S_WIDTH + 128) and (pos3[1] > 452 and pos3[1] < 477):
                   prevind = index
                   index = (index-1)%len(pictures)
                   for i in range(0,len(todraw)):
                       pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                   draw_buttons(surface,6)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
               elif (pos3[0] > 548 and pos3[0] < S_WIDTH + 128) and (pos3[1] > 34 and pos3[1] < 74):
                   for i in range(0,len(todraw)):
                        pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                        screen.blit(surface, (0,0))
                   img = pygame.surfarray.pixels3d(screen)
                   img = img[20:S_WIDTH+20,20:S_HEIGHT+20,:]
                   img = ndimage.rotate(img,90)
                   img = img[::-1,:]
                   draw_buttons(surface,7)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()
                   if len(pictures)>0:
                       plt.imsave(f"{dir_save}/{pictures[index]}", img)
                       with open(f"{txtsave}/{txt[index]}", 'w') as f:
                           for i in range(len(todraw)):
                               f.write(f"{int((todraw[i][0]-19)*width/S_WIDTH)}\t{int((todraw[i][1]-19)*height/S_HEIGHT)}\t{int(0.5+rad[i]*width/S_WIDTH)}\n")
                   prevind = index
                   index = (index+ 1)%len(pictures)
                   draw_buttons(surface,100)
                   text_render(surface, surface)
                   screen.blit(surface, (0,0))
                   pygame.display.update()

       if pos != None:
           if event.type == pygame.MOUSEMOTION:
                    posx = event.pos
                    for i in range(0,len(todraw)):
                            pygame.draw.circle(surface, clr1, (todraw[i]), int(rad[i]),bor_size)
                            screen.blit(surface, (0,0))
                    r = pygame.Rect((20, 20),(S_WIDTH, S_HEIGHT))
                    radius = sqrt((pos[0] - posx[0])**2 + (pos[1] - posx[1])**2)//2
                    pygame.draw.rect(surface, clr2, r1)
                    pygame.draw.rect(surface, clr2, r2)
                    pygame.draw.rect(surface, clr2, r3)
                    pygame.draw.rect(surface, clr2, r4)
                    screen.blit(surface, (0,0))
                    pygame.draw.circle(surface, clr1, (int((pos[0]+posx[0])/2), int((pos[1]+posx[1])/2)), int(radius), bor_size)
                    screen.blit(surface, (0,0))
                    draw_buttons(surface,1)
                    pygame.draw.rect(surface, clr2, r1)
                    pygame.draw.rect(surface, clr2, r2)
                    pygame.draw.rect(surface, clr2, r3)
                    pygame.draw.rect(surface, clr2, r4)
                    draw_buttons(surface,8)
                    text_render(surface, surface)
                    screen.blit(surface, (0,0))
                    pygame.display.update()
    return pos, todraw, rad, redo, redorad, index, prevind, txt, save_index


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
#       print(pic2[-3:])
       if (pic2[:-4] not in test) and (pic2[-3:] == "jpg" or pic2[-3:] == "png"):
           txtfiles.append(f"{pic2[:-4]}.txt")
  #         print("OK")
       elif pic2[:-5] not in test:
 #          print("OK2")
           txtfiles.append(f"{pic2[:-5]}.txt")
   txtfiles = sorted(txtfiles)
#   print(txtfiles)
   for txtf in txtfiles:
       if txtf[-4:] == ".csv" or txtf[-4:] == ".txt":
           txt2.append(txtf)
  # print(txt2)
   for t in txt2:
       if t[:-3] in test_jpg or t[:-3] in test_jpeg:
           txt3.append(t)
   print(f" The images you have in the directory are {pic}")
   print(f" The txt files you have in the directory are {txt3}")
   return pic, txt3
