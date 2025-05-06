from greedy import test_greedy
from dqn_test import test_dqn
from test_model import test_model
import numpy as np
import statistics
import time

def plot_results():
    # np.random.seed(0)
    start_time = time.time()
    results = test_model()
    elapsed_time = time.time() - start_time
    num_tests = len(results)
    microservices_in_nodes = []
    total_requests = []
    delays = []
    congestions = []
    costs = []
    stored_in_edge = []
    stored_in_cloud = []
    collocated = []
    parallelism_ratios = []
    
    for result in results:
        parallelism_ratios.append(statistics.mean(result.parallelism_ratio))
        collocated.append(result.collocated_tasks/result.total_ms)
        total_requests.append(result.current_app_total)
        # print("Number of Requests:", total_requests[-1])
        allocation_per_node = np.vstack((np.vstack(result.allocation_per_timeslot_domain), result.allocation_per_timeslot_shared))
        microservices_in_nodes.append(allocation_per_node)
        congestions.append(result.congestion_occurences)
        delays.append(statistics.mean(result.app_total_comp_times))
        # consumptions.append(statistics.mean(result.operating_costs))
        costs.append(statistics.mean(result.power_consumptions))
        stored_in_edge.append(result.stored_in_edge/result.total_ms)
        stored_in_cloud.append(result.stored_in_cloud/result.total_ms)
        # print(result.total_ms)
    print("Mean App Total Completion Time: ", statistics.mean(delays))
    # print("Average Processing Cost per Microservice: ", statistics.mean(consumptions))
    print("Average Power Consumption per Microservice: ", statistics.mean(costs))
    print("Percentage Stored in Edge: ", statistics.mean(stored_in_edge))
    print("Percentage Stored in Cloud: ", statistics.mean(stored_in_cloud))
    print("Average Execution Time: " + str(elapsed_time/len(results)) + " seconds.")
    print("Average Congestion Occurrences: ", statistics.mean(congestions))
    print("Percentage of collocated tasks: ", statistics.mean(collocated))
    print("Average Parallelism Ratio: ", statistics.mean(parallelism_ratios))

    # # Iterate over each node and its data
    # for nodes_data in microservices_in_nodes:
    #     microservices_per_node = []
    #     for node in nodes_data:
    #         node_microservices = set()
    #         for time_slot in node:
    #             for entry in time_slot:
    #                 # Extracting the microservice number (4th element of each tuple)
    #                 microservice_number = entry
    #                 node_microservices.add(microservice_number)
    #         microservices_per_node.append(node_microservices)
    #     for node_set in microservices_per_node:
    #         print(len(node_set))
    
if __name__ == '__main__':

    plot_results()