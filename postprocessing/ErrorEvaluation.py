import numpy as np


def calculateError(j_sims, j_index, positive_indices, steps=100):
    sim = 0.365
    TP,TN,FP,FN = 0,0,0,0
    space = np.linspace(j_sims[-1],j_sims[0],steps)
    for i in range(len(j_index)):
        if j_sims[i] >= sim:
            if j_index[i] in positives:
                TP += 1
            else:
                FP += 1
        else:
            if j_index[i] in positives:
                FN += 1
            else:
                TN += 1
    precision = TP/(TP+FP)
    recall = (TP/(TP+FN))
    accuracy = ((TP + TN)/(TP + TN + FP + FN))
    print("recall,", "precision,", "accuracy,", recall, precision, accuracy)
    print("TP, FP, FN, TN", TP, FP, FN, TN)
