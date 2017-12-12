import numpy as np

def do_LDA(class_1_data, class_0_data):
    total_data = np.vstack((class_1_data, class_0_data))
    dim = len(class_1_data[0])
    n1 = len(class_1_data)
    n2 = len(class_0_data)
    print total_data
    mean = []
    mean.append(np.mean(class_1_data,axis=0))
    mean.append(np.mean(class_0_data,axis=0))
    mean = np.vstack((mean[0],mean[1]))
    
    one_scatter_within = np.zeros((dim,dim), dtype=float)
    for row in(class_1_data):
        row, mv = row.reshape(dim,1), mean[0].reshape(2,1)
        one_scatter_within = one_scatter_within + (row - mv).dot((row - mv).T)
        
    zero_scatter_within = np.zeros((dim,dim), dtype=float)
    for row in(class_0_data):
        row, mv = row.reshape(dim,1), mean[1].reshape(dim,1)
        zero_scatter_within = zero_scatter_within + (row - mv).dot((row - mv).T)
        
    scatter_within = one_scatter_within + zero_scatter_within
    
    complete_mean = np.mean(mean, axis=0).reshape(dim,1)
    one_mean = mean[0].reshape(dim,1)
    one_scatter_between = n1*(one_mean - complete_mean).dot((one_mean - complete_mean).T)
    zero_mean = mean[1].reshape(dim,1)
    zero_scatter_between = n2*(zero_mean - complete_mean).dot((zero_mean - complete_mean).T)
    scatter_between = one_scatter_between + zero_scatter_between
    
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(np.array(scatter_within))
    .dot(np.array(scatter_between)))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    
    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
   
    
    W = eig_pairs[0][1].reshape(2,1)
    
    Y = total_data.dot(W.real)
    return Y
    


