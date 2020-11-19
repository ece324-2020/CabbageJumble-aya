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
    path_to_imgs = "../../../Final_images_subset", #"/Users/yvielcastillejos/Desktop/To be labelled 1-images" ,
    path_to_label = "../../../reall_final_labels_subset", #"/Users/yvielcastillejos/Desktop/To be labelled 1-labels 2",

    # path to save labelled image and txt label respectively:
    path_to_save_label = "../../../FINAL_final_labels",
    border_size = 3
	
)

keys = {
	# -1 stands for no label (Don't change this! Can only change 0-9)

	# FOR LABEL 1 (CLASS OF COINS)
        # Stores 0,1,2,3,4,5 in txt file
        -1: ' ',
	0: "$0.01",
	1: "$0.05",
	2: "$0.10",
	3: "$0.25",
	4: "$1.00",
	5: "$2.00",
        6: " ",
        7: " ",
        8: " ",
        9: " ",

	# FOR LABEL 2, PRESS "h" OR "t"
        # Stores the ASCII number in the txt file
	72: "H", # HEADS
	84: "T" # TAILS
}


Screen = dict(
    # can change to a square
    S_WIDTH = 800,
    S_HEIGHT = 800,
    GRIDSIZE = 50
)

colour = dict(
    # clr 1 is the colour of the shape
    clr2 = (0, 0,0),
    clr1 = (255, 7, 58),
    clr3 = (57,255,20),

    selected = ((96, 108, 118)),
    unselected = (193, 200, 199)
)


