import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labels_path', type=str, required = True)
parser.add_argument('--fixed_labels_path', type=str, required = True)

#py change_v0_v1_labels.py --labels_path FINAL_final_labels --fixed_labels_path corrected_FINAL_final_labels 

args = parser.parse_args()

labels = sorted(os.listdir(args.labels_path))

mapping = {}
mapping['0'] = "1"
mapping['1'] = "5"
mapping['2'] = "10"
mapping['3'] = "25"
mapping['4'] = "100"
mapping['5'] = "200"


for i in labels:
    name = i

    f = open(f"{args.labels_path}/{name}","r")
    f2 = open(f"{args.fixed_labels_path}/{name}","w")

    lines = f.readlines()
    for j in lines:
        line = j.split("\t")
        key = line[-2]
        coin_value = mapping[key]
        line[-2] = coin_value

        string = "\t".join(line)
        f2.write(string)

    f.close()
    f2.close()

    
