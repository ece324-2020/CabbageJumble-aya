import pygame 
import config

# path
dir = config.misc['path_to_imgs']
dir_save =  config.misc['path_to_save']
txtfolder = config.misc['path_to_label']
txtsave = config.misc['path_to_save_label']
if txtfolder == "" or txtfolder == " " or txtfolder == None:
    txtfolder = dir
if txtsave == "" or txtsave == " " or txtsave == None:
    txtsave = dir_save


# Screen
S_WIDTH = config.Screen['S_WIDTH']
S_HEIGHT = config.Screen['S_HEIGHT']
GRIDSIZE = config.Screen['GRIDSIZE']

# Circle radius and colour:
clr1 = config.colour['clr1']
bor_size = config.misc["border_size"]

# colour
clr2 = config.colour['clr2']
selected = config.colour['selected']
unselected = config.colour['unselected']

# render text
def text_render(surface, screen):
        myfont = pygame.font.SysFont("fontname", 15)
        text2 = myfont.render("'save' and 'next':", 1, (0,0,0))
        screen.blit(text2, (S_WIDTH + 50,37))
        text2 = myfont.render("(spacebar)", 1, (0,0,0))
        screen.blit(text2, (S_WIDTH + 62,57))
        text1 = myfont.render("Save", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,117))
        text1 = myfont.render("(Enter)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,127))
        text1 = myfont.render("Undo (d)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,217))
        text1 = myfont.render("Redo (r)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,317))
        text1 = myfont.render("Next (>)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 68,407))
        text1 = myfont.render("Previous (<)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 68,457))
        myfont = pygame.font.SysFont("fontname", 20)
        text1 = myfont.render("Press 'c' to clear (Can be undone by pressing 'r')", 1, (0,0,0))
        screen.blit(text1, (20 ,0))
        return

# render buttons:
def draw_buttons(surface,x):
   r10 = pygame.Rect(((S_WIDTH + 48,34),(GRIDSIZE+35, GRIDSIZE-10)))
   pygame.draw.rect(surface,  selected, r10,6)
   pygame.draw.rect(surface, unselected, r10)
   r = pygame.Rect(((S_WIDTH + 68,102),(GRIDSIZE, GRIDSIZE)))
   pygame.draw.rect(surface, unselected, r)
   pygame.draw.rect(surface, selected, r,5)
   r1 = pygame.Rect(((S_WIDTH + 68,202),(GRIDSIZE, GRIDSIZE)))
   pygame.draw.rect(surface, unselected, r1)
   pygame.draw.rect(surface, selected, r1,5)
   r2 = pygame.Rect(((S_WIDTH + 68,302),(GRIDSIZE, GRIDSIZE)))
   pygame.draw.rect(surface, unselected, r2)
   pygame.draw.rect(surface, selected , r2,5)
   r3 = pygame.Rect(((S_WIDTH + 58,402),(GRIDSIZE+25, GRIDSIZE-25)))
   pygame.draw.rect(surface, unselected, r3)
   pygame.draw.rect(surface, selected , r3,5)
   r4 = pygame.Rect(((S_WIDTH + 58,452),(GRIDSIZE+25, GRIDSIZE-25)))
   pygame.draw.rect(surface, unselected, r4)
   pygame.draw.rect(surface, selected , r4,5)
   margin = pygame.Rect((20,20),(S_WIDTH, S_HEIGHT))
   pygame.draw.rect(surface, (10,10,10) , margin,3)

   pygame.draw.rect(surface, clr2, r10,2)
   pygame.draw.rect(surface, clr2, r,2)
   pygame.draw.rect(surface, clr2, r1,2)
   pygame.draw.rect(surface, clr2, r2,2)
   pygame.draw.rect(surface, clr2, r3,2)
   pygame.draw.rect(surface, clr2, r4,2)
   if x == 1:
          pygame.draw.rect(surface, selected, r)
   if x == 3:
          pygame.draw.rect(surface, selected, r1)
   if x == 4:
          pygame.draw.rect(surface, selected, r2)
   if x == 5:
          pygame.draw.rect(surface, selected, r3)
   if x == 6:
          pygame.draw.rect(surface, selected, r4)
   if x == 7:
          pygame.draw.rect(surface, selected, r10)
   return

