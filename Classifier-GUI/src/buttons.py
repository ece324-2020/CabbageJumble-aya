import pygame 
import config

# path
dir = config.misc['path_to_imgs']
dir_save = "idk" # config.misc['path_to_save']
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
clr3 = config.colour['clr3']
selected = config.colour['selected']
unselected = config.colour['unselected']

# render text
def text_render(surface, screen, label1, label2, todraw):
        myfont = pygame.font.SysFont("fontname", 20)
        text2 = myfont.render("'save' and 'next':", 1, (0,0,0))
        screen.blit(text2, (S_WIDTH + 50,37))
        text2 = myfont.render("(spacebar)", 1, (0,0,0))
        screen.blit(text2, (S_WIDTH + 62,57))
        text1 = myfont.render("Save", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,117))
        text1 = myfont.render("(Enter)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 75,127))
        text1 = myfont.render("Next (>)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 68,407))
        text1 = myfont.render("Previous (<)", 1, (0,0,0))
        screen.blit(text1, (S_WIDTH + 68,457))
        myfont = pygame.font.SysFont("fontname", 20)

        for i in range(len(todraw)):
               myfont = pygame.font.SysFont("fontname", 15)
               text1 = myfont.render(f"{config.keys[label1[i]]}, {config.keys[label2[i]]}", 1, (57,255,20))
               surface.blit(text1, (todraw[i][0]-17, todraw[i][1]-8))
        pygame.display.update()       
        return

# render buttons:
def draw_buttons(surface,x):
    return
