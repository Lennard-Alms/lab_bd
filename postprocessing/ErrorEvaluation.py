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


def jacc_sim_calc(H, Q):
    M = H * Q[np.newaxis,:]
    intersection = M.sum(axis=1)
    union = H.sum(axis=1) + Q.sum() - intersection
    j_sim = intersection/union
    indices = np.argsort(-j_sim)
    similarities = j_sim[indices]
    return indices, similarities

def hamming_distance(H, Q):
    xor = np.logical_xor(H, Q[np.newaxis,:])
    distances = xor.sum(axis=1)
    return distances

def hamming_similarity(H, Q):
    distances = hamming_distance(H, Q)
    similarities = distances / H.shape[1]
    indices = np.argsort(-similarities)
    similarities = similarities[indices]
    return similarities, indices
