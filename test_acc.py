import random
import numpy as np
import matplotlib.pyplot as plt

file = open('./output_data.txt','r')
line = file.readline()
count = 0
total = 0
while(line) :
    line = line.split()
    gt = line[0]

    predict = line[1]
    gt_array = gt.split("_")
    gt_name = gt_array[0]
    if len(gt_array) == 3:
        gt_name += "_" + gt_array[1]
    if(gt_name == predict):
        count = count + 1
    
    total =total +1

    line = file.readline()

print(count / total)

file.close()