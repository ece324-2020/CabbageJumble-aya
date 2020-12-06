# Bounding_Shape_Tool
A tool useful for bounding circles around an object. This tool can be used for bounding images for Neural Network Training.              
## Tips:
- Change the paths in ```config.py```
- The images are recommended to be squares before passing into the tool to avoid distortion (but can be any size)
- The original image will not be affected, but screenshots will be made incase that may be useful.
- txt files are generated with the pixel x and y coordinates of the circle and the radius. (These coordinates are with respect to the original image shape)
- The GUI will automatically find a txt file with the same name as the image (if the path is correct) and will automatically draw the circles if there is     information in the txt files.
- One can change the screen width and height in config.py and it will work fine.
- Supported images are jpg, jpeg, and png
- txt files are generated with the pixel x and y coordinates of the circle and the radius. (These coordinates are with respect to the original image shape)
- One can now modify the circles at any time (One can leave the GUI and continue at another time)
- Adjust the border size and colour of circle in  ```config.py```
- I would recommend making  ```path_to_label``` and ```path_to_label_save``` the same path for convenience.

## Instructions
If one wants to use this tool:
- Clone this repository
- Install pygame library
- Install math libraries such as Numpy, Scipy, and Matplotlib.
- In ```config.py```, change the paths
- Note that making the directory ```path_to_img``` and ```path_to_save``` the same path, the original image will be overwritten by the labelled image. (I recommend against this)
- Similarly, making ```path_to_label``` and ```path_to_label_save```  the same path will overwrite the txt files (I recommend making them the same path)
- run   ``` python3 src/main.py```


| Files| Description|
|-----|-------------|
| ```main.py```| run this to use the tool|
| ```config.py``` | contains the configuration for the program. Change file path here.|
|```boundbox.py```| contains the functions for renderring GUI|
|```buttons.py ```| Functions for rendering the buttons and text|

## Demo
A demo as shown:                                
<img src = "https://github.com/yvielcastillejos/Bounding_Shape_Tool/blob/main/images/updated2.gif" width = "500" height = "500">

## Txt File
The txt file contains the x,y and radius coordinates of the circle with respect to the original image size. This is the "label" part when training a Neural Network and it also serves as a checkpoint for the GUI so that the user can continue modifying the circles at any time.                     
<img src = "https://github.com/yvielcastillejos/Bounding_Shape_Tool/blob/main/images/Screen%20Shot%202020-11-08%20at%206.08.32%20PM.png" width = "300" height = "300">
## Next Steps
- I can add the option to use bounding squares instead of circles.
- This was inspired by coin counter project.
