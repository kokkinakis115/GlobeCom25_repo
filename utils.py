import numpy as np
import random

## Helper Functions

def create_DAG(num_of_ms):
    dag_matrix = np.zeros((num_of_ms, num_of_ms), dtype=np.int16)
    for i in range(num_of_ms):
        for j in range(i, num_of_ms):
            if (i == j):
                dag_matrix[i][j] = 0
            else:
                dag_matrix[i][j] = np.random.choice([0, 1], p=[0.4, 0.6])
                dag_matrix[j][i] = 0
    return dag_matrix 

def unique_values(g):
    s = set()
    for x in g:
        if x in s: return False
        s.add(x)
    return True

def floyd_warshall(graph):
    n = len(graph)
    # stepping loop
    for k in range(n):
        # outer loop
        for i in range(n):
            # inner loop
            for j in range(n):
                # replace direct path with path through k if direct path is longer
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
    return graph

def create_adj_matrix(num_of_nodes, edge_nodes, fog_nodes, cloud_adj_matrix): #creates adjacency matrix for graph
    # print(cloud_adj_matrix)
    edge = (0, edge_nodes)
    fog = (edge_nodes, edge_nodes+fog_nodes)
    matrix = np.zeros((num_of_nodes, num_of_nodes), dtype=np.int16)
    for i in range(edge[0], edge[1]):
        for j in range(i, edge[1]):
            if (i == j):
                matrix[i][j] = 0
            else:
                matrix[i][j] = np.random.randint(1,10)
                matrix[j][i] = matrix[i][j]
        for j in range(edge[1], fog[1]):
            matrix[i][j] = np.random.randint(10,50)
            matrix[j][i] = matrix[i][j]
            
    for i in range(fog[0], fog[1]):
        for j in range(i, fog[1]):
            # print(i, j)
            if (i == j):
                matrix[i][j] = 0
            else:
                matrix[i][j] = cloud_adj_matrix[i-fog[0]][j-fog[0]]
                matrix[j][i] = matrix[i][j]                        

    # adj_matrix = floyd_warshall(matrix)
    # return adj_matrix
    return matrix

def reward_func(node_percentage, latency, cost, weights):
    reward = weights['num_of_fragments']*node_percentage - weights['cost']*cost - weights['latency']*latency
    return reward

def reward_func_cmplx(node_percentage, src_latency, trgt_latency, cost, weights):
    reward = weights['num_of_fragments']*node_percentage - weights['cost']*cost - weights['store_latency']*src_latency - weights["retrieve_latency"]*trgt_latency
    return reward
    
def normalize(values, min_val, max_val):
    return (values - min_val) / (max_val - min_val)


def generate_poisson_events(rate, time_duration):
    num_events = np.random.poisson(rate * time_duration)
    event_times = np.sort(np.random.uniform(0, time_duration, num_events))
    return num_events, event_times

def generate_request(num_ms, max_ms):
    data_size = np.random.choice([20, 30, 50, 100], p=[0.35, 0.3, 0.3, 0.05])
    microservice_ids = np.random.randint(0, num_ms, max_ms)
    dependencies = create_DAG(max_ms)
    return (data_size, microservice_ids, dependencies)

def find_parents_adj_matrix(adj_matrix, target_node):
    # Find all nodes that have an edge pointing to the target node (column in matrix)
    parents = [i for i in range(target_node) if adj_matrix[i][target_node] == 1]
    return parents

def create_joint_matrix(matrices, n_exclusive, n_shared):
    k = len(matrices)  # Number of matrices
    total_nodes = k * n_exclusive + n_shared
    
    # Initialize the joint matrix
    joint_matrix = np.zeros((total_nodes, total_nodes))
    
    # Add exclusive and shared parts for each matrix
    for i, matrix in enumerate(matrices):
        # Exclusive part
        start = i * n_exclusive
        end = start + n_exclusive
        joint_matrix[start:end, start:end] = matrix[:n_exclusive, :n_exclusive]
        
        # Shared part: Add connections between exclusive and shared nodes
        joint_matrix[start:end, -n_shared:] = matrix[:n_exclusive, -n_shared:]
        joint_matrix[-n_shared:, start:end] = matrix[-n_shared:, :n_exclusive]
    
    # Add shared nodes connections (they are the same in all matrices)
    joint_matrix[-n_shared:, -n_shared:] = matrices[0][-n_shared:, -n_shared:]

    return joint_matrix
    