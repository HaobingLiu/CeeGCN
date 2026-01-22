import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def FCM(data, n_clusters, m):

    max_iter = 100
    tol = 1e-5

    if isinstance(data, torch.Tensor):
        data = data.detach()


    data = data.cpu().numpy() 

    centroids = np.random.rand(n_clusters, data.shape[1])

    membership_matrix = np.random.rand(data.shape[0], n_clusters)
    membership_matrix /= np.sum(membership_matrix, axis=1)[:, None]
   
    


    for _ in range(max_iter):
        inner_product = np.dot(data, centroids.T)

        new_membership_matrix = 1 / np.power(inner_product, 2 / (m - 1))
        new_membership_matrix /= np.sum(new_membership_matrix, axis=1)[:, None]

        centroids = np.dot(new_membership_matrix.T, data) / np.sum(new_membership_matrix, axis=0)[:, None]

        if np.linalg.norm(new_membership_matrix - membership_matrix) < tol:
            break

        membership_matrix = new_membership_matrix

    return membership_matrix, centroids
