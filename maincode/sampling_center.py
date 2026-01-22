import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def read_graph(file_path):
    data = pd.read_csv(file_path, header=None, names=['node1', 'node2', 'weight'])
    nodes = np.unique(data[['node1', 'node2']].values)
    return data, nodes

def build_kd_tree(nodes):
    return KDTree(nodes.reshape(-1, 1))

def calculate_density_weights(data, nodes):
    density = {node: 0 for node in nodes}
    for _, row in data.iterrows():
        density[row['node1']] += row['weight']
        density[row['node2']] += row['weight']
    return density

def select_initial_seed(density):
    return max(density, key=density.get)

def calculate_distance_weights(kd_tree, seed, nodes):
    seed_array = np.array([[seed]])
    distances, _ = kd_tree.query(seed_array, k=len(nodes))
    distances = distances[0]  
    distance_weights = {node: distances[i] for i, node in enumerate(nodes)}
    return distance_weights


def exclude_noise_points(density, threshold):
    return {node: d for node, d in density.items() if d >= threshold}


def select_seed(density, distance_weights, selected_seeds):
    max_weight = -1
    next_seed = None
    for node, density_weight in density.items():
        if node not in selected_seeds:
            weight = density_weight - distance_weights[node]
            if weight > max_weight:
                max_weight = weight
                next_seed = node
    return next_seed


def rdbi_clustering(file_path, k):
    data, nodes = read_graph(file_path)
    kd_tree = build_kd_tree(nodes)
    density = calculate_density_weights(data, nodes)
    initial_seed = select_initial_seed(density)

    clusters = [initial_seed]
    selected_seeds = {initial_seed}
    noise_threshold = np.median(list(density.values())) 

    density = exclude_noise_points(density, noise_threshold)

    for _ in range(k - 1):
        distance_weights = calculate_distance_weights(kd_tree, initial_seed, nodes)
        next_seed = select_seed(density, distance_weights, selected_seeds)
        if next_seed:
            clusters.append(next_seed)
            selected_seeds.add(next_seed)
    return clusters

def sample_center():
    file_path = '/home/graph-cluster/movie/boat/boat_gra.csv'
    k = 9
    centers = rdbi_clustering(file_path, k)
    return centers