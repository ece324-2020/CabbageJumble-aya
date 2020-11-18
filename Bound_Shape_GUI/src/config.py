# Parameters

''' You need a path to the images "path_to_imgs" (after the OpenCV segmentation), 

    a path to the txt files "path_to_label" ( also after the segmentation),
   		 If there are no txt files, you can point it to an empty folder.
                 also if some of the images are missing txt files, thats fine,
                 the gui will just generate it for you if you draw circles.

    "path_to_save" are screenshots of the images with bounded circles (they are useless for us)
                 you can just point it to an empty folder and delete afterwards


    "path_to_save_label" this one is the path where the modified txt files are stored. 
                         If you make it the same  path as  "path_to_label", it will 
                         modify the original txt files by appending/removing stuff. 
                         Also, the txtfiles will be named the same 
                         as the image it corresponds to (disregarding the extension)

   you can delete the bounded circles in three ways: one you can press "d", two you can delete manually in txt file
   and three, you can press clear or "c". Note that the circles will be automatically drawn if there are txt files that correspond to the images
   (and you can modify these circles at any time by adding to it or removing it)


Note that in order to save the circles, you have to press spacebar or enter to save them
'''

########## PARAMETERS ############



''' I really recommend making path_to_label and path_to_save_label the same path so that way
    you would know whether an image has been labelled or not (e.g. you can be going in circles without knowing it)
    (It's just more convenient). Then you can just manually move the labelled images and modified txt files 
    onto the good folder, also the txt files will have the same name as the image it corresponds to '''

misc = dict(

    # path to your folder of images and txt files respectively (if they are in the same folder, put the same path):
    # If you don't have any txt files yet, just point it to an empty folder
    path_to_imgs = "../../Data_augmentation/Debug_augmented_images" ,
    path_to_label ="../../Data_augmentation/Debug_augmented_labels",

    # path to save labelled image and txt label respectively:
    path_to_save = "../../Data_augmentation/temp_folder_test_GUI_images" ,
    path_to_save_label = "../../Data_augmentation/temp_folder_test_GUI_labels",
    border_size = 5
)

Screen = dict(
    # can change to a square
    S_WIDTH = 600,
    S_HEIGHT = 600,
    GRIDSIZE = 50
)

colour = dict(
    # clr 1 is the colour of the shape
    clr2 = (193, 200, 199),
    clr1 = (255,0,0),

    selected = ((96, 108, 118)),
    unselected = (193, 200, 199)
)


