# CabbageJumble-aya

## Introduction
Comedically inspired by the thought of paying with a wheelbarrow of coins, Cabbage Jumble-aya’s goal is to count the value of coins in a pile, given only an image. 

## Functionality and Limitations
Cabbage Jumble-aya counts coins given a picture

### 

## Major Roadblocks!
-[ ] Baseline segmenter breaks for touching coins/ coins partially outside view
-[ ] Labelling areas and training data
-[ ] 

## Work Plan, Deadlines, and Work Flow
|Date|Yviel|David|James|
|---|---|---|---|
|2020 Nov 9|Baseline CNN, GUI to label|Finish segmentation baseline, break code into libraries|Label areas that are part of coins|
|2020 Nov 16|Modifying GUI to label value and Heads/Tails|Convert JSON to txt, clean up coding library|Labelling values/Heads/Tails|

## Baseline Model
<img src = "https://github.com/ece324-2020/CabbageJumble-aya/blob/main/Images/baseline.png" width = "1000" height = "300">


## Final Model
<img src = "https://github.com/ece324-2020/CabbageJumble-aya/blob/main/Images/Full_arc.png" width = "1000" height = "300">

## Example 


### Tools Used
- Bounding GUI: https://github.com/ece324-2020/CabbageJumble-aya/tree/main/Bound_Shape_GUI
- Labelling GUIS: https://github.com/ece324-2020/CabbageJumble-aya/tree/main/Classifier-GUI/src

### Data Augmentation Techniques
<img src = "https://github.com/ece324-2020/CabbageJumble-aya/blob/main/Images/DataProc.png" width = "1000" height = "300">

### Accuracy and Results
<img src =  "https://github.com/ece324-2020/CabbageJumble-aya/blob/main/Images/Example.png" width = "500" height = "500">


### References and Acknowledgement
- Source Code 1: https://github.com/Tianxiaomo/pytorch-YOLOv4
- Source code 2: https://github.com/AlexeyAB/darknet
- Paper Yolo v4: https://arxiv.org/abs/2004.10934

