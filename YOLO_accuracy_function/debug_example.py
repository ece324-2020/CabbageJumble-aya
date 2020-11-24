##this section is used to Debug the accuracy function
'''
[[0.3812929764389992, 0.7121105529367924, 0.17229157779365778, 0.16927140182815492, 0.9892747, 0.9881557, 4], [0.6601750431582332, 0.648126058280468, 0.16360190883278847, 0.1676964859943837, 0.98070073, 1.0, 5], [0.5238729901611805, 0.44300566567108035, 0.15318832360208035, 0.14369975426234305, 0.87046885, 0.97317106, 1], [0.39947509206831455, 0.2558608492836356, 0.1425792621448636, 0.15032200957648456, 0.63271195, 0.9988281, 1]]


250,245,312,308,5 130,273,198,336,4 191,156,255,213,1 147,81,210,140,1

'''

import torch
from acc_func_v2_working import *

YOLO_labels = [[0.3812929764389992, 0.7121105529367924, 0.17229157779365778, 0.16927140182815492, 0.9892747, 0.9881557, 4], [0.6601750431582332, 0.648126058280468, 0.16360190883278847, 0.1676964859943837, 0.98070073, 1.0, 5], [0.5238729901611805, 0.44300566567108035, 0.15318832360208035, 0.14369975426234305, 0.87046885, 0.97317106, 1], [0.39947509206831455, 0.2558608492836356, 0.1425792621448636, 0.15032200957648456, 0.63271195, 0.9988281, 1]]
YOLO_labels = torch.FloatTensor(YOLO_labels)

GT_labels = [[250,245,312,308,5],[130,273,198,336,4],[191,156,255,213,1],[147,81,210,140,1]]
GT_labels = torch.FloatTensor(GT_labels)

YOLO_labels = torch.reshape(YOLO_labels,(1,YOLO_labels.shape[0],YOLO_labels.shape[1]))
GT_labels = torch.reshape(GT_labels,(1,GT_labels.shape[0],GT_labels.shape[1]))



print(accuracy_of_images_in_batch(YOLO_labels,GT_labels))

#return segmentation_accuracy_acc,classification_accuracy_acc , pred_vs_GT_acc


#250,245,312,308,5 
#130,273,198,336,4 
#191,156,255,213,1 
#147,81,210,140,1