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

def cosine_distance(H, Q):
    H_length = np.linalg.norm(H, axis=1)
    Q_length = np.linalg.norm(Q, axis=0)
    distances = np.dot(H, Q) / (H_length * Q_length)
    distances[np.where(distances > 1)] = 1
    distances[np.where(distances < -1)] = -1
    return np.arccos(distances)

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

#SC = Same Class Label as Query Label
#IC = Inter Class Label, class of patch > 0
def evaluate_result(result_ids, class_labels, query_label):
  if isinstance(class_labels, list):
    class_labels = np.array(class_labels)
  SCTP, SCTN, SCFP, SCFN = 0,0,0,0
  ICTP, ICTN, ICFP, ICFN = 0,0,0,0

  number_of_images = len(class_labels)
  sc_true_positives = []
  ic_true_positives = []

  for index in result_ids:
    if class_labels[index] == query_label:
      SCTP += 1
    if class_labels[index] != query_label:
      SCFP += 1
    if class_labels[index] > 0:
      ICTP += 1
    if class_labels[index] == 0:
      ICFP += 1

  SC_SUM = (class_labels == query_label).sum()
  IC_SUM = (class_labels > 0).sum()

  SCFN = SC_SUM - SCTP
  ICFN = IC_SUM - ICTP
  SCTN = number_of_images - SCTP - SCFP - SCFN
  ICTN = number_of_images - ICTP - ICFP - ICFN

  sc_precision = SCTP / (SCTP + SCFP)
  ic_precision = ICTP / (ICTP + ICFP)

  sc_recall = SCTP / (SCTP + SCFN)

  ic_recall = ICTP / (ICTP + ICFN)

  sc_accuracy = (SCTP + SCTN) / (number_of_images)
  ic_accuracy = (ICTP + ICTN) / (number_of_images)

  return ((sc_precision, sc_recall, sc_accuracy), (ic_precision, ic_recall, ic_accuracy))
