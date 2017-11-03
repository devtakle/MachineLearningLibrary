# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA


def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]




def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
def shouldStop(oldCentroids, centroids):
    return np.all(oldCentroids == centroids)

    
def k_means(data,initial_centroids, k):
    centroids = initial_centroids 
    oldCentroids = centroids.copy()
    closest = closest_centroid(data, centroids)
    centroids = move_centroids(data, closest, centroids)
    iterations = 1
    while shouldStop(oldCentroids, centroids) == False:
        oldCentroids = centroids
        closest = closest_centroid(data, centroids)
        centroids = move_centroids(data, closest, centroids)
        iterations = iterations + 1
    
    return closest, iterations
    
"""new_x = dev_pca.do_PCA(X, 2)
k_means(new_x,3)"""
        
    
          

