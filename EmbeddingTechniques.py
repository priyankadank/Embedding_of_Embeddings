#!/usr/bin/python
import pymf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import manifold, decomposition
import scipy.spatial.distance as dist



class EmbeddingTechniques():
    def __init__(self, datasetname):
        self.decomposition_methods={
             'CUR':pymf.CUR,
             'SVD':pymf.SVD,
             'PCA':pymf.PCA,
             'NMF':pymf.NMF,
             'SIVM':pymf.SIVM,
             'Kmeans':pymf.Kmeans,
             'AA':pymf.AA,
             'Isomap':manifold.Isomap,
             'LLE':manifold.LocallyLinearEmbedding,
             'FastICA':decomposition.FastICA,
             
        }

        
    def normalize_variance(self,datasetname):    #applies Mean-shift & divide by variance normalization
        
        data_from_dataset = np.genfromtxt(datasetname, delimiter=',', skip_header=1, dtype=float)
        means = np.mean(data_from_dataset, axis=0)
        shifted_data = data_from_dataset - means
        variances = np.std(shifted_data, axis=0)
        self.data = np.array(shifted_data/variances)
        
    def normalize_dataminmax(self,datasetname):	  #applies min-max normalization	
        data_from_dataset = np.genfromtxt(datasetname, delimiter=',', skip_header=1, dtype=float)
        data_min = np.min(data_from_dataset, axis=0)
        data_max = np.max(data_from_dataset, axis=0)
        shifted_data = data_from_dataset - data_min
        self.data = shifted_data/(data_max)

    def without_normalization(self,datasetname):  #without normalization
        self.data = np.genfromtxt(datasetname, delimiter=',', skip_header=1, dtype=float)
        
    def get_factors(self, method,n_neighbors=4,n_components=2,num_iterations=10, num_bases=20,show_progress=False):
        data=self.data
        print method.__name__
        if method.__name__ =='Isomap' or method.__name__ == 'FastICA' or method.__name__ =='LocallyLinearEmbedding':
            m = method(n_components=n_components)
            m.fit(data)
            return m.transform(data),method.__name__
        else:
            try:
                m = method(data, compW=True, num_bases=num_bases, niter=num_iterations)
                m.initialization()
                m.factorize()
                return m.W, method.__name__
            except:
                m = method(data, show_progress=show_progress, rrank=num_bases)
                m.factorize()
                A  = np.dot(m.U, m.S)
                return A,method.__name__
                if method.__name__ == "SVD":
                    A = A[:,:2]
                    return A, method.__name__

    def get_neighbors(self,A,description): 
        neigh = KNeighborsClassifier(n_neighbors=3)#n_neighbors are no. of neighbors to be considered
        neigh.fit(A, None)
        N = neigh.kneighbors_graph(A).todense()
        N = np.array(N)#put neighborhood relation into matrix.if neighbor then 1 else 0.
        #print "N=",N
        
        neigh_set = []
        for a in N:
            neighs = np.where(a==1)[0]	# diagonal elements will be 1 as each element is neighbor of itself  
            neigh_set.append(set(neighs))

        #print neigh_set
        
        d={description:neigh_set}
        return d
        

        


        

