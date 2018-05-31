#!/usr/bin/python
from EmbeddingTechniques import EmbeddingTechniques
import pylab as pl
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition


d = {}
from glob import glob
all_datasets = glob("datasets\\*.csv")
accmulated_similarities = np.zeros((10,10))
for datasetname in all_datasets:
    picture=EmbeddingTechniques(datasetname)
    picture.normalize_variance(datasetname) #apply Mean-shift & divide by variance normalization
    #picture.normalize_dataminmax(datasetname)	#apply Min-max normalization
    #picture.without_normalization(datasetname)	#without normalization
    print "datasetname=",datasetname
    for description, method in picture.decomposition_methods.items():
        A,method_name=picture.get_factors(method, n_neighbors=4,n_components=2,num_iterations=50, num_bases=2,show_progress=False)
        print "\n", description, A.shape
#         pl.clf()
#         pl.title(description)
#         pl.scatter(A.T[0], A.T[1])
#         pl.show()			#gives the output for each of the embedding technique
        d.update(picture.get_neighbors(A,description))	#finds the k neighbors of each of the datapoint and stores in the dictionary
        #print "d=",d.keys()
    P = []
    Q = []
    sim = []
    k = d.keys()
    m = np.zeros((len(k), len(k)))
    for x in range(0,len(k)):
        for y in range(x+1,len(k)):
            try:
                P = np.asarray(d.get(k[x]))
                Q = np.asarray(d.get(k[y]))
                sim = []
                for i in range(len(P)):
                    sim.append(float(len(P[i].intersection(Q[i])))/float(len(P[i].union(Q[i]))))
                m[x,y] = sum(sim)/len(sim)
                m[y,x] = sum(sim)/len(sim)
                accmulated_similarities[x,y] += sum(sim)/len(sim)	#calculates pairwise similarity index 
                accmulated_similarities[y,x] += sum(sim)/len(sim)	#calculates pairwise similarity index 
            except:
                pass
    accmulated_similarities += np.eye(len(k))
accmulated_similarities /= float(len(all_datasets))	#calculates average similarity index
d.clear()
fig = pl.figure()
ax = pl.imshow(accmulated_similarities, cmap=pl.cm.hot, interpolation="nearest", vmin=0.0, vmax=1.0)
pl.xticks(range(len(k)), k, rotation = 90)
pl.yticks(range(len(k)), k)
pl.colorbar()
pl.title("Similarity graph")
pl.show()		#outputs pairwise similarity graph
    
mds = manifold.MDS(n_components=2)
mds.fit(accmulated_similarities)
e = mds.fit_transform(accmulated_similarities)
fig=pl.figure()
ax = fig.add_subplot(111)
pl.title("Embedding of embeddings")
for i,x in enumerate(e):
    pl.annotate(k[i], xy=e[i])
ax.scatter(e.T[0], e.T[1]) #T is transpose  
pl.show()			#plots 2d embedding of embeddings graph


    






