# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
import operator 

#the function ensures zero empirical mean, standardises data

def standardise(array):
    for i in range(len(array[0])):
        empMean = np.mean(array[:,i])
        array[:,i] = (array[:,i] - empMean)/np.std(array[:,i])
       
        
def do_PCA(X, components, do_std= True):
    if(do_std):
        standardise(X)
        
    cov_mat = np.cov(X.T)
    ei_vals, ei_vecs = np.linalg.eig(cov_mat)

    #print ei_vecs
    #coverting to tuples (eigenvalues, eigenvectors)
    ei_pairs = [(np.abs(ei_vals[i]), ei_vecs[:,i]) for i in range(len(ei_vals))]
    ei_pairs = sorted(ei_pairs, key=operator.itemgetter(0))
   
    ei_pairs.reverse()
    #from this we can tell how many principal components to choose
    #it gives us the cumulative variance starting from the first PC
    tot = sum(ei_vals)
    var_exp = [(i / tot)*100 for i in sorted(ei_vals, reverse=True)]
    for i in ei_pairs:
        cumulative = np.cumsum(var_exp)
    print "cumulative variance given by principal components"
    print cumulative 
    #getting final Y
    new_ei_vecs = np.vstack(x[1] for x in ei_pairs).T
    Y = np.dot(X,new_ei_vecs[:,:components])
    return Y

#read data from csv into pandas list
#orgX = pd.read_csv('dataset.csv')

#individual list for variables
"""x = orgX.x
y = orgX.y
z = orgX.z
X = np.stack((x,y,z),axis = -1) 
Y = do_PCA(X, 2)
PCA_result = sklPCA(n_components=2).fit_transform(X)
print Y
print PCA_result
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Y[:,0], Y[:,1])
plt.show()"""

