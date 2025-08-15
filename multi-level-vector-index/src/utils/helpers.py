import os
import tempfile
import numpy as np

def pad_to_k(arr, k, pad_value):
    arr = list(arr)
    if len(arr) < k:
        arr += [pad_value] * (k - len(arr))
    return arr[:k]

def calculate_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def flatten_nested_dict(nested_dict):
    flat_dict = {}
    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            flat_dict[(outer_key, inner_key)] = value
    return flat_dict

def sort_by_distance(metadata):
    return sorted(metadata, key=lambda x: x[0])  # Sort by distance

def get_cluster_centroids(kmeans):
    return kmeans.centroids if hasattr(kmeans, 'centroids') else None

def create_cache_path(name, url):
    return os.path.join(tempfile.gettempdir(), url.split('/')[-1])