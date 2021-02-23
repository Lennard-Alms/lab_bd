import numpy as np


def calculateError(_sims, _index, positive_indices, steps=100):
    space = np.linspace(0,1,steps)
    precisions = []
    accuracies = []
    recalls = []
    for sim in space:
        TP,TN,FP,FN = 0,0,0,0
        for i in range(len(_index)):
            if _sims[i] >= sim:
                if _index[i] in positive_indices:
                    TP += 1
                else:
                    FP += 1
            else:
                if _index[i] in positive_indices:
                    FN += 1
                else:
                    TN += 1
        TN = len(_sims) - len(positive_indices)
        TP = len(positive_indices)
        precision = TP/(TP+FP)
        recall = (TP/(TP+FN))
        accuracy = ((TP + TN)/(TP + TN + FP + FN))
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    return recalls, precisions, accuracies, space
    # print("recall,", "precision,", "accuracy,", recall, precision, accuracy)
    # print("TP, FP, FN, TN", TP, FP, FN, TN)

def jaccard_similarity(H, Q, sort=True):
    M = H * Q[np.newaxis,:]
    intersection = M.sum(axis=1)
    union = H.sum(axis=1) + Q.sum() - intersection
    similarities = intersection/union
    if sort:
        indices = np.argsort(-similarities)
        similarities = similarities[indices]
    else:
        indices = np.arange(H.shape[0])
    return similarities, indices

def cosine_distance(H, Q, normalized=False, via_hash=False):
    if via_hash:
        return hamming_distance(H, Q) * np.pi / H.shape[1]
    if normalized:
        return np.arccos(np.dot(H,Q[np.newaxis,:]).flatten())
    else:
        return np.arccos(np.dot(H,Q[np.newaxis,:]).flatten() / (np.linalg.norm(H, axis=1) * np.linalg.norm(Q)))

def sort_by_distance(distances):
    indices = np.argsort(distances)
    return distances[indices], indices

def hamming_distance(H, Q):
    xor = np.logical_xor(H, Q[np.newaxis,:])
    distances = xor.sum(axis=1)
    return distances

def hamming_similarity(H, Q, sort=True):
    distances = hamming_distance(H, Q)
    similarities = 1 - distances / H.shape[1]
    if sort:
        indices = np.argsort(-similarities)
        similarities = similarities[indices]
    else:
        indices = np.arange(H.shape[0])
    return similarities, indices
