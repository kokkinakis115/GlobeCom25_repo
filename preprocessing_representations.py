import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data
import statistics
import numpy as np

# def produce_graph_repr_worker(latencies, node_capacities, node_costs, device_coef, hosted_microservices, current_app_allocation, latency_from_user, power_consumption):

#     num_nodes = len(node_costs)
#     # Node Characteristics preperation 
#     node_features = []
#     for i in range(num_nodes):
#         # print(type([node_costs[i]]), type(node_capacities[i].tolist()), type(hosted_microservices[i].tolist()), type([device_coef[i]]))
#         feature_vector = [node_costs[i]] + [statistics.mean(node_capacities[i])] + hosted_microservices[i].tolist() + [device_coef[i]] + [1 if i in current_app_allocation else 0] + [latency_from_user[i]] + [power_consumption[i]] 
#         # print(len(feature_vector))
#         node_features.append(feature_vector)

#     # Edge Characteristics preperation
#     edge_index = []
#     edge_weights = []
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i != j and latencies[i][j] > 0:
#                 edge_index.append([i, j])
#                 edge_weights.append(latencies[i][j])
            
#     node_features = torch.tensor(node_features, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t()
#     edge_weights = torch.tensor(edge_weights, dtype=torch.float)

#     graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

#     return graph_data

# def produce_app_repr(dependencies, current_ms, num_microservices):

#     num_ms = dependencies.shape[0]

#     microservice_id = one_hot(torch.Tensor(microservice_ids).to(torch.int64), num_classes=num_microservices)
#     dependencies = torch.tensor(dependencies)
#     # Node Characteristics preperation 
#     source_nodes, target_nodes = torch.where(dependencies == 1)
#     edge_index = torch.stack([source_nodes, target_nodes], dim=0)

#     node_features = []
#     for i in range(num_ms):
#         feature_vector = microservice_id.tolist()[i]+[microservice_cpu[microservice_ids[i]]]+[microservice_startup[microservice_ids[i]]]+[1 if i == current_ms else 0] # size: 10+1+1+1+1+1
#         # print(feature_vector)
#         node_features.append(feature_vector)
#     # print(node_features)
#     node_features = torch.tensor(node_features, dtype=torch.float32)

#     graph_data = Data(x=node_features, edge_index=edge_index)
#     # print(graph_data.x)

#     return graph_data

def produce_app_repr(task_features, dependencies, current_ms, num_microservices, num_dependencies):

    task_features = task_features[:current_ms+1]  # Select features for the current microservices
    dependencies = dependencies[:num_dependencies]
    dependencies = [[entry[0], entry[1]] for entry in dependencies if entry[0] <= current_ms and entry[1] <= current_ms]  # Filter dependencies for the current microservices
    if dependencies == []:
        dependencies = [[0, 0]]
    current_indicator = torch.zeros((current_ms+1, 1), dtype=torch.float32)
    current_indicator[current_ms] = 1.0
    node_features = torch.tensor(task_features, dtype=torch.float32)
    node_features = torch.cat((node_features, current_indicator), dim=1)  # Concatenate along the feature dimension

    edge_index = torch.tensor(dependencies, dtype=torch.long).t()

    data = Data(x=node_features, edge_index=edge_index)
    return data

def produce_local_app_repr(task_features, dependencies, current_ms, num_microservices, num_dependencies):
    """
    Creates a local subgraph Data object for the current_ms node and its direct neighbors.

    Parameters:
        task_features (list or np.ndarray): List of task feature vectors (topo-ordered).
        dependencies (list of tuples): List of edges as (src, dst).
        current_ms (int): Index of the current microservice node.

    Returns:
        torch_geometric.data.Data object with only current_ms and its neighbors.
    """

    current_ms = int(current_ms)
    # Step 1: Build edge index from dependencies
    dependencies = [[src, dst] for src, dst in dependencies]

    # Step 2: Find neighbors of current_ms
    neighbors = set()
    for src, dst in dependencies:
        if src == current_ms or dst == current_ms:
            neighbors.add(src)
            neighbors.add(dst)
    neighbors.add(current_ms)
    neighbors = sorted(list(neighbors))  # Ensure consistent order
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbors)}

    # Step 3: Filter and remap edges
    local_edges = [
        [node_mapping[src], node_mapping[dst]]
        for src, dst in dependencies
        if src in node_mapping and dst in node_mapping and (src == current_ms or dst == current_ms)
    ]

    if not local_edges:
        local_edges = [[0, 0]]  # Dummy edge to avoid empty edge_index

    edge_index = torch.tensor(local_edges, dtype=torch.long).t()

    # Step 4: Slice features and create current_indicator
    node_features_np = np.array([task_features[i] for i in neighbors], dtype=np.float32)
    node_features = torch.from_numpy(node_features_np)
    current_indicator = torch.zeros((len(neighbors), 1), dtype=torch.float32)
    current_indicator[node_mapping[current_ms]] = 1.0

    node_features = torch.cat((node_features, current_indicator), dim=1)

    # Step 5: Create Data object
    data = Data(x=node_features, edge_index=edge_index)

    return data


def produce_infr_repr(min_capacities, node_costs, device_coef, latencies, current_app_allocation, predecessors):
    num_nodes = latencies.shape[0]
    pred_allocated_indicator = np.zeros(num_nodes, dtype=np.float32)
    # has_last_request = np.zeros(num_nodes, dtype=np.float32)
    for task_idx in predecessors:
        assigned_node = current_app_allocation[task_idx]
        if assigned_node >= 0:  # Check if it has been allocated
            pred_allocated_indicator[assigned_node] = 1.0
    node_features = np.stack([min_capacities, node_costs, device_coef, pred_allocated_indicator], axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if latencies[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append([latencies[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

def produce_infr_repr_mod(avg_capacities ,min_capacities, node_costs, device_coef, latencies, current_app_allocation, predecessors):
    num_nodes = latencies.shape[0]
    pred_allocated_indicator = np.zeros(num_nodes, dtype=np.float32)
    # has_last_request = np.zeros(num_nodes, dtype=np.float32)
    for task_idx in predecessors:
        assigned_node = current_app_allocation[task_idx]
        if assigned_node >= 0:  # Check if it has been allocated
            pred_allocated_indicator[assigned_node] = 1.0
    node_features = np.stack([avg_capacities, min_capacities, node_costs, device_coef, pred_allocated_indicator], axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if latencies[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append([latencies[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data

