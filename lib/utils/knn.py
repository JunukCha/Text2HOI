import numpy as np
from scipy.spatial.distance import cdist

import torch

def knn_numpy(x, k=3):
    """
    Compute KNN for points in x using scipy's cdist function.
    
    Parameters:
        x (np.ndarray): A numpy array of shape (point_num, feature_dim).
        k (int): The number of nearest neighbors to find.
        
    Returns:
        np.ndarray: A numpy array of shape (point_num, k) containing the indices of k nearest neighbors.
    """
    # Compute pairwise distance matrix using cdist
    distance_matrix = cdist(x, x, metric='euclidean')
    
    # Sort distance matrix and pick k smallest values
    indices = np.argsort(distance_matrix, axis=1)
    knn_indices = indices[:, :k]
    
    return knn_indices

def knn_numpy_matrix(x, k=3):
    """
    Compute KNN for points in x using scipy's cdist function.
    
    Parameters:
        x (np.ndarray): A numpy array of shape (point_num, feature_dim).
        k (int): The number of nearest neighbors to find.
        
    Returns:
        np.ndarray: A numpy array of shape (point_num, k) containing the indices of k nearest neighbors.
    """
    point_num = x.shape[0]

    # Compute pairwise distance matrix using cdist
    distance_matrix = cdist(x, x, metric='euclidean')
    
    # Sort distance matrix and pick k smallest values
    indices = np.argsort(distance_matrix, axis=1)
    knn_indices = indices[:, :k]
    
    # Create a binary matrix
    binary_matrix = np.zeros((point_num, point_num), dtype=int)
    
    row_indices = np.repeat(np.arange(point_num), k)
    binary_matrix[row_indices, knn_indices.flatten()] = 1
    return binary_matrix

def knn_pytorch(x, k=3):
    """
    Compute KNN for the batch of points in x.
    
    Parameters:
        x (torch.Tensor): A tensor of shape (batch_size, point_num, feature_dim).
        k (int): The number of nearest neighbors to find.
        
    Returns:
        torch.Tensor: A tensor of shape (batch_size, point_num, k) containing the indices of k nearest neighbors.
    """
    # Compute pairwise distance matrix
    distance_matrix = torch.cdist(x, x, p=2)
    # inner_product = torch.matmul(x, x.transpose(2, 1))
    # square_norm = inner_product.diagonal(dim1=1, dim2=2).unsqueeze(1)
    # distance_matrix = square_norm + square_norm.transpose(1, 2) - 2 * inner_product
    
    # Sort distance matrix and pick k smallest values
    _, indices = torch.sort(distance_matrix, dim=2)
    knn_indices = indices[:, :, :k]
    
    return knn_indices