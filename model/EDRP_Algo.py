import numpy as np
import math

class EDRP_Algo:
    def __init__(self, n, d1=0.5, d2=0.7):
        p1 = 1 - d1 / np.pi
        p2 = 1 - d2 / np.pi
        k = int(math.ceil(- np.log2(n) / np.log2(p2)))
        roh = np.log2(p1) / np.log2(p2)
        L = int(np.ceil((n ** roh) / p1))
        delta = 1/np.e
        print('With probability ' + str(round(1-delta,2)) + ' this will return at least one epsilon nearest neighbour if one exists.')
        self.d2 = d2
        self.k = k
        self.L = L
        self.signature_size = k * L

    def create_hash_tables(self, hashes):
        self.T = []
        for l in range(self.L):
          t = {}
          for p in range(hashes.shape[0]):
            hash_key = str( hashes[ p, self.k * l : self.k * (l+1) ] )
            if hash_key in t:
              t[hash_key].append(p)
            else:
              t[hash_key] = [p]
          self.T.append(t)

    def query_for(self, query):
        colisions = set()
        for l in range(self.L):
          hash_key = str( query[ self.k * l : self.k * (l+1) ] )
          if hash_key in self.T[l]:
              for p in self.T[l][hash_key]:
                colisions.add(p)
        return list(colisions)



























    #
