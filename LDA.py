import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
data = pd.read_csv('SCLC_study_output_filtered_2.csv')
dataM = data.as_matrix()

class_n_data = dataM[:20,1:]
class_s_data = dataM[20:,1:]
total_data = dataM[:,1:]
mean = []
mean.append(np.mean(class_n_data,axis=0))
mean.append(np.mean(class_s_data,axis=0))
mean = np.vstack((mean[0],mean[1]))

n_scatter_within = np.zeros((19,19), dtype=float)
for row in(class_n_data):
    row, mv = row.reshape(19,1), mean[0].reshape(19,1)
    n_scatter_within = n_scatter_within + (row - mv).dot((row - mv).T)
    
s_scatter_within = np.zeros((19,19), dtype=float)
for row in(class_s_data):
    row, mv = row.reshape(19,1), mean[1].reshape(19,1)
    s_scatter_within = s_scatter_within + (row - mv).dot((row - mv).T)
    
scatter_within = n_scatter_within + s_scatter_within

complete_mean = np.mean(mean, axis=0).reshape(19,1)
n_mean = mean[0].reshape(19,1)
n_scatter_between = 20*(n_mean - complete_mean).dot((n_mean - complete_mean).T)
s_mean = mean[1].reshape(19,1)
s_scatter_between = 20*(s_mean - complete_mean).dot((s_mean - complete_mean).T)
scatter_between = n_scatter_between + s_scatter_between

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(np.array(scatter_within, dtype=float)).dot(np.array(scatter_between, dtype=float)))
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    
"""first eigenvector explains 100% of the variance"""

W = eig_pairs[0][1].reshape(19,1)

Y = total_data.dot(W.real)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(Y[:20,0].T, len(Y[:20,0].T) * [1], color='blue', label='N')
ax.scatter(Y[20:,0].T, len(Y[20:,0].T) * [1], color='red', label='S')
ax.legend(loc='upper left')
fig.suptitle("own LDA implementation")
fig.show()

"""use SKLearn LDA to compare"""
target_N = np.ones(20)
target_S = np.zeros(20)
target = np.hstack((target_N, target_S))
LDA = skLDA(solver = 'eigen')
LDA.fit(dataM[:,1:], target)
output = LDA.transform(dataM[:,1:])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(output[:20,0].T, len(output[:20,0].T) * [1], color='blue', label='N')
ax.scatter(output[20:,0].T, len(output[20:,0].T) * [1], color='red', label='S')
ax.legend(loc='upper left')
fig.suptitle("SKLearn LDA implementation")
fig.show()
"""comparing values we get a mean scale of 1 suggesting own LDA perfectly 
matches SKLearn LDA"""
print "own output/SKLearn output gives us"
print np.mean(Y/output)*100 ,"% match"
