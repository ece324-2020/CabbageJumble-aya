#py check_txt_format.py --path corrected_FINAL_final_labels 

'''
The path is a folder of text files and prints one of these text files.
You can change if count == 2: break line to print more than 1 txt files. It currently prints 2 txt files.

Helpful to see what the format is like in a txt file.
'''


import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required = True)

args = parser.parse_args()

labels = sorted(os.listdir(args.path))

count = 0
for i in labels:
    name = i

    f = open(f"{args.path}/{name}","r")
    lines = f.readlines()
    print(lines)
    f.close()
    count+=1
    #
    if count == 2:
        break

    
