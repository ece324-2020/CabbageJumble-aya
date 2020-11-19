import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required = True)

#py check_txt_format.py --path corrected_FINAL_final_labels 

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
    if count == 2:
        break

    
