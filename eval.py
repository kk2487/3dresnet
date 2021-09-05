import random
import numpy as np
import matplotlib.pyplot as plt
def eval(input,GT) :
    print('ip = ', input)
    print('GT = ', GT)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    npos = 0
    precision = np.zeros(len(input))
    recall = np.zeros(len(input))
    for i in range(len(input)):
        if (input[i] == 1 and GT[i] == 1):
            TP += 1
            npos += 1
        elif (input[i] == 0 and GT[i] == 0):
            TN += 1
        elif (input[i] == 1 and GT[i] == 0):
            FP += 1
        elif (input[i] == 0 and GT[i] == 1):
            FN += 1
            npos += 1
        if (TP != 0 or FP != 0):
            precision[i] = float(TP / (TP + FP))
            # print('i = %d,TP = %d,FN = %d,TP+FN = %d,TP/(TP+FN) = %f'%(i+1,TP,FN,(TP+FN),float(TP/(TP+FN))))
        else:
            precision[i] = 1
        if (TP != 0):
            recall[i] = float(TP)
            # print('i = %d,TP = %d,FN = %d,TP+FN = %d,TP/(TP+FN) = %f'%(i+1,TP,FN,(TP+FN),float(TP/(TP+FN))))
        else:
            recall[i] = 0
        # print('i = %d,TP = %d,FP = %d,precision = %f,recall = %f'%(i,TP,FP,precision[i],recall[i]))
    recall = recall / npos
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    if npos == 0:
        print('Waring no positive instances')
    return(ap,precision,recall)
red = open('/home/rvl/KaiChun/CEO/Eye_Classfication/output_data.txt','r')
line = red.readline()
normal_in = []
normal_gt = []
blink_in = []
blink_gt = []
while(line) :
    line = line.split()
    name = line[0]
    gt = line[1]
    if name[0:5] == 'blink' :
        blink_in += [1]
        normal_in +=[ 0]
    else :
        blink_in += [0]
        normal_in += [1]
    if gt == 'blink' :
        blink_gt += [1]
        normal_gt += [0]
    else :
        blink_gt += [0]
        normal_gt += [1]
    line = red.readline()
ap,pre,rec = eval(blink_in,blink_gt)
print('ap = ',ap)
print('pre = ',pre[len(pre)-1])
print('rec = ',rec[len(rec)-1])

ap,pre,rec = eval(normal_in,normal_gt)
print('ap = ',ap)
print('pre = ',pre[len(pre)-1])
print('rec = ',rec[len(rec)-1])


