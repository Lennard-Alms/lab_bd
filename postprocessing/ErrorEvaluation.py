import numpy as np


def calculateError(_sims, _index, positive_indices, steps=100):
    space = np.linspace(_sims[-1],_sims[0],steps)
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

        precision = TP/(TP+FP)
        recall = (TP/(TP+FN))
        accuracy = ((TP + TN)/(TP + TN + FP + FN))
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    return recalls, precisions, accuracies, space
    # print("recall,", "precision,", "accuracy,", recall, precision, accuracy)
    # print("TP, FP, FN, TN", TP, FP, FN, TN)


def jaccard_similarity(H, Q):
    M = H * Q[np.newaxis,:]
    intersection = M.sum(axis=1)
    union = H.sum(axis=1) + Q.sum() - intersection
    j_sim = intersection/union
    indices = np.argsort(-j_sim)
    similarities = j_sim[indices]
    return similarities, indices

def cosine_distance(H, Q):
    H_length = np.linalg.norm(H, axis=1)
    Q_length = np.linalg.norm(Q, axis=0)
    distances = np.dot(H, Q) / (H_length * Q_length)
    distances[np.where(distances > 1)] = 1
    distances[np.where(distances < -1)] = -1
    return np.arccos()

def hamming_distance(H, Q):
    xor = np.logical_xor(H, Q[np.newaxis,:])
    distances = xor.sum(axis=1)
    return distances

def hamming_similarity(H, Q):
    distances = hamming_distance(H, Q)
    similarities = 1 - distances / H.shape[1]
    indices = np.argsort(-similarities)
    similarities = similarities[indices]
    return similarities, indices
