import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import pickle
import utils
import networkx as nx
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os

base_path = os.path.dirname(os.path.abspath(__file__))
class Environment(MultiAgentEnv):
    metadata = {'render.modes': ['console']}

    def __init__(self, params):
        super(Environment, self).__init__()

        self.time_periods = params.get('time_periods', 2)
        self.nr_agents = params.get('agents', 1)
        self.num_nodes_domain  = params.get('num_nodes_domain', 2)
        self.num_nodes_shared = params.get('num_nodes_shared', 2)
        self.capacity_range_domain  = params.get('capacity_range_domain', (20, 50))
        self.capacity_range_shared = params.get('capacity_range_shared', (200, 500))
        self.num_microservices = params.get('num_microservices', 20)
        self.request_arrival_rate = params.get('arrival_rate', 5)
        self.look_ahead_window = params.get('look_ahead_window', 50)
        self.weights = params.get('weights', {'utilization': 1, 'cost': 1, 'latency': 1, 'power': 1})
        self.max_ms = params.get('max_ms', 5)
        self.window = params.get('window', 20)
        self.max_tasks = params.get('max_tasks', 20)
        self.task_features = params.get('task_features', 2)
        self.max_dependencies = params.get('max_dependencies', 50)
        self.from_trace = params.get('from_trace', True)
        
        self.current_period = 0
        self.current_app_total = 0
        self.current_app = np.zeros(self.nr_agents, dtype=np.int32)
        self.current_ms = np.zeros(self.nr_agents, dtype=np.int32)
        self.current_app_endtimes = [{} for agent in range(self.nr_agents)]
        self.current_app_allocation = (-1)*np.ones((self.nr_agents, self.max_tasks), dtype=np.int32)
        self.app_total_comp_times = []
        self.operating_costs = []
        self.power_consumptions = []
        self.stored_in_edge = 0
        self.stored_in_cloud = 0
        self.total_ms = 0
        self.no_op_action = self.num_nodes_domain + self.num_nodes_shared
        # self.trace_path = os.path.join(base_path, './pkl_files/job_graphs_avg_20.93_tasks.pkl')
        self.trace_path = os.path.join(base_path, './job_graphs_cleaned.pkl')
        # print("Getting traces from: ", self.trace_path)
        self.congestion_occurences = 0
        self.parallelism_ratio = []
        self.collocated_tasks = 0
        self.app_spans = np.zeros(self.nr_agents, dtype=np.int32)

        # Node Characteristics
        # Capacities
        self.total_node_capacity_domain = np.random.randint(self.capacity_range_domain[0], self.capacity_range_domain[1], size=(self.nr_agents, self.num_nodes_domain))
        self.total_node_capacity_shared = np.random.randint(self.capacity_range_shared[0], self.capacity_range_shared[1], size=self.num_nodes_shared)
        # self.max_val = max(self.total_node_capacity_domain.max(), self.total_node_capacity_shared.max())
        self.max_val = self.capacity_range_shared[1]

        # Power
        self.power_consumption_domain = np.random.uniform(1, 5, (self.nr_agents, self.num_nodes_domain))
        # self.max_cost = self.node_costs_domain.max()
        self.max_consumption = 20
        self.power_consumption_shared = np.random.uniform(10, 20, self.num_nodes_shared)
        
        self.node_capacities_domain = np.array([[[float(self.total_node_capacity_domain[agent][node]) for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_domain)] for agent in range(self.nr_agents)])
        self.node_capacities_shared = np.array([[float(self.total_node_capacity_shared[node]) for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_shared)])
        self.remaining_node_capacities_domain = self.node_capacities_domain.copy()
        self.remaining_node_capacities_shared = self.node_capacities_shared.copy()
        
        # Costs
        self.node_costs_domain = np.random.uniform(3, 5, (self.nr_agents, self.num_nodes_domain))
        # self.max_cost = self.node_costs_domain.max()
        self.max_cost = 5
        self.node_costs_shared = np.random.uniform(1, 3, self.num_nodes_shared)
        self.operating_costs = []
        
        # Device Coefficient
        self.device_coef_domain = np.random.choice([1, 1.5, 2 , 5], (self.nr_agents, self.num_nodes_domain), p = [0.1, 0.4, 0.4, 0.1])
        self.device_coef_shared = np.random.choice([0.25, 0.5, 0.75, 1], self.num_nodes_shared, p = [0.1, 0.4, 0.4, 0.1])
        # Hosted Ms
        self.hosted_microservices_domain = np.random.choice([0, 1], (self.nr_agents, self.num_nodes_domain, self.num_microservices), p = [0.4, 0.6])
        self.hosted_microservices_shared = np.random.choice([0, 1], (self.num_nodes_shared, self.num_microservices), p = [0.2, 0.8])
        
        self.latency_from_user = np.concatenate((np.random.randint(1, 5, self.nr_agents), np.array([15])))
        self.latencies_domain = np.random.randint(1, 5, (self.nr_agents, self.num_nodes_domain, self.num_nodes_domain))

        latencies_shared = np.random.randint(20,40 ,size=(self.num_nodes_shared, self.num_nodes_shared))
        latencies_shared = (latencies_shared + latencies_shared.T)/2
        self.latencies = np.array([utils.create_adj_matrix(self.num_nodes_domain+self.num_nodes_shared, self.num_nodes_domain, self.num_nodes_shared, latencies_shared) for _ in range(self.nr_agents)])
        
        self.joint_latencies = utils.create_joint_matrix(self.latencies, self.num_nodes_domain, self.num_nodes_shared)
        self.max_latency = 40
        self.allocation_per_timeslot_domain = np.array([[[set() for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_domain)] for agent in range(self.nr_agents)])
        self.allocation_per_timeslot_shared = np.array([[set() for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_shared)])


        # Application Characteristics
        if self.from_trace:
            with open(self.trace_path, 'rb') as f:
                job_graphs = pickle.load(f)
                job_list = list(job_graphs.values())
        # self.microservice_cpu = np.random.uniform(0.1, 5, self.num_microservices)
        # self.microservice_startup = np.random.choice([5, 10, 25], self.num_microservices)
        requests_per_agent = []
        requests_to_schedule_per_agent= []
        for _ in range(self.nr_agents):
            num_requests , event_times = utils.generate_poisson_events(self.request_arrival_rate, self.time_periods)
            num_requests_per_period = [np.where(np.logical_and(event_times>time_period, event_times<=time_period+1))[0].shape[0] for time_period in range(self.time_periods)]
            requests = []
            for period in range(self.time_periods):
                request=[(np.random.choice([20, 30, 50, 100], p=[0.35, 0.3, 0.3, 0.05]), random.choice(job_list)) for _ in range(num_requests_per_period[period])]
                requests.append(request)
            requests_to_schedule_per_agent.append(num_requests_per_period.copy())
            requests_per_agent.append(requests)
        self.requests_per_agent = requests_per_agent
        self.requests_to_schedule_per_agent = np.array(requests_to_schedule_per_agent)
        

        self.agents = self.possible_agents = [f"agent_{i}" for i in range(self.nr_agents)]
        self.observation_spaces = {
            agent: spaces.Dict({
                "latencies": spaces.Box(low=0, high=100, shape=(self.num_nodes_domain+self.num_nodes_shared,self.num_nodes_domain+self.num_nodes_shared), dtype=np.int16),
                "node_capacities": spaces.Box(low=0, high=500, shape=(self.num_nodes_domain+self.num_nodes_shared,self.look_ahead_window), dtype=np.float32),
                "node_costs": spaces.Box(low=0, high=5, shape=(self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                "power_consumption": spaces.Box(low=0, high=20, shape=(self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                "device_coef": spaces.Box(low=0, high=2, shape=(self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                "request_features": spaces.Box(low=-100, high=1000, shape=(self.max_tasks, self.task_features), dtype=np.float32),
                "request_dependencies": spaces.Box(low=-1, high=self.max_tasks, shape=(self.max_dependencies, 2), dtype=np.int32),
                "num_tasks": spaces.Discrete(self.max_tasks+1),
                "num_dependencies": spaces.Discrete(self.max_dependencies+1),
                "current_app": spaces.Discrete(2*self.request_arrival_rate+1),
                "current_ms": spaces.Discrete(self.max_tasks),
                "requests_left": spaces.Discrete(self.max_tasks),
                "agent_active": spaces.Discrete(2),
                "current_allocation": spaces.Box(low=-1, high=self.num_nodes_domain+self.num_nodes_shared, shape=(self.max_tasks,), dtype=np.int32)
        }) for agent_index, agent in enumerate(self.agents)
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.num_nodes_domain+self.num_nodes_shared+1) for agent in self.agents
        }

    # def get_joint_observation(self):
    #     return {
    #         "latencies": self.joint_latencies,
    #         "node_capacities": np.vstack((np.vstack(self.remaining_node_capacities_domain[:, :, self.current_period:self.current_period+self.look_ahead_window]), self.remaining_node_capacities_shared[:, self.current_period:self.current_period+self.look_ahead_window])),
    #         "node_costs": np.concatenate([self.node_costs_domain.flatten(), self.node_costs_shared.flatten()]),
    #         "power_consumption": np.concatenate([self.power_consumption_domain.flatten(), self.power_consumption_shared.flatten()]),
    #         "device_coef": np.concatenate([self.device_coef_domain.flatten(), self.device_coef_shared.flatten()]),
    #         "hosted_microservices": np.vstack((np.vstack(self.hosted_microservices_domain), self.hosted_microservices_shared)),
    #         "microservice_cpu": self.microservice_cpu,
    #         "microservice_startup": self.microservice_startup,
    #         "current_ms": self.current_ms,
    #         "current_app": self.current_app,
    #         "requests": self.requests_per_agent,
    #         "latencies_from_user": self.latency_from_user_obs
    #         }

    def get_all_observations(self):
        observations = {}
        for agent_index, agent in enumerate(self.agents):
            if self.requests_to_schedule_per_agent[agent_index][self.current_period] - self.current_app[agent_index]== 0:
                observations[agent] = {
                    "latencies": np.zeros((self.num_nodes_domain+self.num_nodes_shared,self.num_nodes_domain+self.num_nodes_shared), dtype=np.int16),
                    "node_capacities": np.zeros((self.num_nodes_domain+self.num_nodes_shared,self.look_ahead_window), dtype=np.float32),
                    "node_costs": np.zeros((self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                    "power_consumption": np.zeros((self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                    "device_coef": np.zeros((self.num_nodes_domain+self.num_nodes_shared,), dtype=np.float32),
                    "request_features": np.zeros((self.max_tasks, self.task_features), dtype=np.float32),
                    "request_dependencies": np.zeros((self.max_dependencies, 2), dtype=np.int32),
                    "num_tasks": 0,
                    "num_dependencies": 0,
                    "current_app": 0,
                    "current_ms": 0,
                    "requests_left": 0,
                    "agent_active": 0,
                    "current_allocation": -1*np.ones(self.max_tasks, dtype=np.int32)
                }
            else:
                observations[agent] = {
                    "latencies": self.latencies[agent_index],
                    "node_capacities": np.vstack((self.remaining_node_capacities_domain[agent_index][:, self.current_period:self.current_period+self.look_ahead_window], self.remaining_node_capacities_shared[:, self.current_period:self.current_period+self.look_ahead_window])),
                    "node_costs": np.concatenate((self.node_costs_domain[agent_index], self.node_costs_shared)),
                    "power_consumption": np.concatenate((self.power_consumption_domain[agent_index], self.power_consumption_shared)),
                    "device_coef": np.concatenate((self.device_coef_domain[agent_index], self.device_coef_shared)),
                    "request_features": self.graph_node_features_array(self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[0][self.current_period]-1)][1], ['cpu_request', 'computation_time'], self.max_tasks),
                    "request_dependencies": self.graph_edges_array(self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[0][self.current_period]-1)][1], self.max_dependencies),
                    "num_tasks": self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1)][1].number_of_nodes(),
                    "num_dependencies": self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1)][1].number_of_edges(),
                    "current_app": min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1),
                    "current_ms": self.current_ms[agent_index],
                    "requests_left": self.requests_to_schedule_per_agent[agent_index][self.current_period]- self.current_app[agent_index],
                    "agent_active": 1 if self.current_app[agent_index] < self.requests_to_schedule_per_agent[agent_index][self.current_period] else 0,
                    "current_allocation": self.current_app_allocation[agent_index]
                }
        return observations

    def reset(self, seed=None, options=None):
        self.current_period = 0
        self.current_app_total = 0
        self.current_app = np.zeros(self.nr_agents, dtype=np.int16)
        self.current_ms = np.zeros(self.nr_agents, dtype=np.int16)
        self.current_app_endtimes = [{} for agent in range(self.nr_agents)]
        self.current_app_allocation = np.zeros((self.nr_agents, self.max_tasks), dtype=np.int32)
        self.app_total_comp_times = []
        self.operating_costs = []
        self.power_consumptions = []
        self.stored_in_edge = 0
        self.stored_in_cloud = 0
        self.total_ms = 0
        self.no_op_action = self.num_nodes_domain + self.num_nodes_shared
        self.congestion_occurences = 0


        # Node Characteristics
        # Capacities
        self.total_node_capacity_domain = np.random.randint(self.capacity_range_domain[0], self.capacity_range_domain[1], size=(self.nr_agents, self.num_nodes_domain))
        self.total_node_capacity_shared = np.random.randint(self.capacity_range_shared[0], self.capacity_range_shared[1], size=self.num_nodes_shared)
        # self.max_val = max(self.total_node_capacity_domain.max(), self.total_node_capacity_shared.max())
        self.max_val = self.capacity_range_shared[1]

        # Power
        self.power_consumption_domain = np.random.uniform(1, 5, (self.nr_agents, self.num_nodes_domain))
        # self.max_cost = self.node_costs_domain.max()
        self.max_consumption = 20
        self.power_consumption_shared = np.random.uniform(10, 20, self.num_nodes_shared)
        
        self.node_capacities_domain = np.array([[[float(self.total_node_capacity_domain[agent][node]) for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_domain)] for agent in range(self.nr_agents)])
        self.node_capacities_shared = np.array([[float(self.total_node_capacity_shared[node]) for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_shared)])
        self.remaining_node_capacities_domain = self.node_capacities_domain.copy()
        self.remaining_node_capacities_shared = self.node_capacities_shared.copy()
        
        # Costs
        self.node_costs_domain = np.random.uniform(3, 5, (self.nr_agents, self.num_nodes_domain))
        # self.max_cost = self.node_costs_domain.max()
        self.max_cost = 5
        self.node_costs_shared = np.random.uniform(1, 3, self.num_nodes_shared)
        self.operating_costs = []
        
        # Device Coefficient
        self.device_coef_domain = np.random.choice([0.75, 1, 1.5], (self.nr_agents, self.num_nodes_domain), p = [0.2, 0.6, 0.2])
        self.device_coef_shared = np.random.choice([0.5, 0.75, 1], self.num_nodes_shared, p = [0.1, 0.4, 0.5])
        # Hosted Ms
        self.hosted_microservices_domain = np.random.choice([0, 1], (self.nr_agents, self.num_nodes_domain, self.num_microservices), p = [0.4, 0.6])
        self.hosted_microservices_shared = np.random.choice([0, 1], (self.num_nodes_shared, self.num_microservices), p = [0.2, 0.8])
        
        self.latency_from_user = np.concatenate((np.random.randint(1, 5, self.nr_agents), np.array([15])))
        self.latencies_domain = np.random.randint(1, 5, (self.nr_agents, self.num_nodes_domain, self.num_nodes_domain))

        latencies_shared = np.random.randint(20,40 ,size=(self.num_nodes_shared, self.num_nodes_shared))
        latencies_shared = (latencies_shared + latencies_shared.T)/2
        self.latencies = np.array([utils.create_adj_matrix(self.num_nodes_domain+self.num_nodes_shared, self.num_nodes_domain, self.num_nodes_shared, latencies_shared) for _ in range(self.nr_agents)])
        
        self.joint_latencies = utils.create_joint_matrix(self.latencies, self.num_nodes_domain, self.num_nodes_shared)
        self.max_latency = 40
        self.allocation_per_timeslot_domain = np.array([[[set() for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_domain)] for agent in range(self.nr_agents)])
        self.allocation_per_timeslot_shared = np.array([[set() for _ in range(self.time_periods+max(self.time_periods//5, 5000))] for node in range(self.num_nodes_shared)])


        # Application Characteristics
        if self.from_trace:
            with open(self.trace_path, 'rb') as f:
                job_graphs = pickle.load(f)
                job_list = list(job_graphs.values())
        # self.microservice_cpu = np.random.uniform(0.1, 5, self.num_microservices)
        # self.microservice_startup = np.random.choice([5, 10, 25], self.num_microservices)
        requests_per_agent = []
        requests_to_schedule_per_agent= []
        for _ in range(self.nr_agents):
            num_requests , event_times = utils.generate_poisson_events(self.request_arrival_rate, self.time_periods)
            num_requests_per_period = [np.where(np.logical_and(event_times>time_period, event_times<=time_period+1))[0].shape[0] for time_period in range(self.time_periods)]
            requests = []
            for period in range(self.time_periods):
                request=[(np.random.choice([20, 30, 50, 100], p=[0.35, 0.3, 0.3, 0.05]), random.choice(job_list)) for _ in range(num_requests_per_period[period])]
                requests.append(request)
            requests_to_schedule_per_agent.append(num_requests_per_period.copy())
            requests_per_agent.append(requests)
        self.requests_per_agent = requests_per_agent
        self.requests_to_schedule_per_agent = np.array(requests_to_schedule_per_agent)

        return self.get_all_observations(), {}
        

    def compute_time_required(self, node_id, agent_id, ms_cpu, data_size):
        if node_id >= self.num_nodes_domain: # compute for shared resources
            node_id = node_id-self.num_nodes_domain
            # ms_id = self.requests_per_agent[agent_id][self.current_period][self.current_app][1][self.current_ms]
            ms_startup = 0
            # data_size = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1]
            time_required = self.device_coef_shared[node_id]*data_size/ms_cpu
        else: # compute for domain resources
            # ms_id = self.requests_per_agent[agent_id][self.current_period][self.[agent_id]][1][self.current_ms]
            # ms_cpu = self.microservice_cpu[ms_id]
            # ms_startup = self.microservice_startup[ms_id]
            ms_startup = 0
            data_size = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][0]
            time_required = self.device_coef_domain[agent_id][node_id]*data_size/ms_cpu/5
        if np.isnan(time_required):
            print(self.device_coef_domain[agent_id][node_id], data_size, ms_cpu)
            # print(time_required)
        return max(time_required, 1)
    
    def get_action_mask(self, agent_id):
        # Initialize the action mask with ones (valid actions)
        task_ordering = list(nx.topological_sort(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1]))
        current_task = task_ordering[self.current_ms[agent_id]]
        action_mask = np.ones(self.num_nodes_domain+self.num_nodes_shared, dtype=np.int32)
        ms_cpu = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes[current_task]['cpu_request']
        data_size = abs(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes[current_task]['computation_time'])
        
        for node in range(self.num_nodes_domain):
            computation_time = int(self.compute_time_required(node, agent_id, ms_cpu, data_size))
            for period in range(computation_time):
                if self.remaining_node_capacities_domain[agent_id][node][self.current_period+period] < ms_cpu:
                    action_mask[node] = 0
                    break
        for node in range(self.num_nodes_shared):
            computation_time = int(self.compute_time_required(node+self.num_nodes_domain, agent_id, ms_cpu, data_size))
            for period in range(computation_time):
                if self.remaining_node_capacities_shared[node][self.current_period+period] < ms_cpu:
                    action_mask[node+self.num_nodes_domain] = 0
                    break
        return action_mask

    def app_is_allocated(self, agent_id):
        return self.current_ms[agent_id] == self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].number_of_nodes()

    def period_is_scheduled(self, agent_id):
        # print(self.requests_to_schedule_per_agent[agent_id][self.current_period], self.current_app)
        return self.requests_to_schedule_per_agent[agent_id][self.current_period] <= self.current_app[agent_id]

    def end_condition(self):
        flag = False
        if self.current_period == self.time_periods-1:
            flag = True
        return flag
    
    def graph_node_features_array(self, G, feature_keys, max_tasks, pad_value=0.0):
        # print(G)
        node_order = list(nx.topological_sort(G))
        feature_matrix = []

        for node in node_order:
            features = [G.nodes[node].get(key, pad_value) for key in feature_keys]
            if features[1] == 0:
                features[1] = 1
            feature_matrix.append(features)

        # Convert to NumPy array
        X = np.array(feature_matrix, dtype=np.float32)

        # Pad if needed
        current_len = X.shape[0]
        num_features = len(feature_keys)

        if current_len < max_tasks:
            pad_rows = np.full((max_tasks - current_len, num_features), pad_value, dtype=np.float32)
            X = np.vstack([X, pad_rows])
        elif current_len > max_tasks:
            X = X[:max_tasks]
            node_order = node_order[:max_tasks]

        return X
    
    def graph_edges_array(self, G, max_edges, pad_value=(-1, -1)):
        # print(G, max_edges)
        # Map node names to indices based on topological order
        node_order = list(nx.topological_sort(G))
        node_index_map = {node: idx for idx, node in enumerate(node_order)}

        edge_list = []
        for u, v in G.edges():
            if u in node_index_map and v in node_index_map:
                edge_list.append((node_index_map[u], node_index_map[v]))

        # Convert to array
        edge_array = np.array(edge_list, dtype=np.int32) if edge_list else np.array([], dtype=np.int32).reshape(0, 2)
        # print(edge_array)
        # Pad or truncate to max_edges
        current_len = edge_array.shape[0]
        if current_len < max_edges:
            pad_rows = np.full((max_edges - current_len, 2), pad_value, dtype=np.int32)
            edge_array = np.vstack([edge_array, pad_rows])
        elif current_len > max_edges:
            edge_array = edge_array[:max_edges]

        return edge_array
    
    
    # def get_observation_space(self, agent_id):
    #     # request = self.requests_per_agent[0][self.current_period][min(self.current_app, self.requests_to_schedule_per_agent[0][self.current_period])-1]
    #     # print(type(request))
    #     agent_index = agent_id
    #     observations = {
    #         "latencies": self.latencies[agent_index],
    #         "node_capacities": np.vstack((self.remaining_node_capacities_domain[agent_index][:, self.current_period:self.current_period+self.look_ahead_window], self.remaining_node_capacities_shared[:, self.current_period:self.current_period+self.look_ahead_window])),
    #         "node_costs": np.concatenate((self.node_costs_domain[agent_index], self.node_costs_shared)),
    #         "power_consumption": np.concatenate((self.power_consumption_domain[agent_index], self.power_consumption_shared)),
    #         "device_coef": np.concatenate((self.device_coef_domain[agent_index], self.device_coef_shared)),
    #         "request_features": self.graph_node_features_array(self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[0][self.current_period])-1][1], ['cpu_request', 'memory_request', 'computation_time'], self.max_tasks),
    #         "request_dependencies": self.graph_edges_array(self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[0][self.current_period])-1][1], self.max_dependencies),
    #         "num_tasks": self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1)][1].number_of_nodes(),
    #         "num_dependencies": self.requests_per_agent[agent_index][self.current_period][min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1)][1].number_of_edges(),
    #         "current_app": min(self.current_app[agent_index], self.requests_to_schedule_per_agent[agent_index][self.current_period]-1),
    #         "current_ms": self.current_ms[agent_index],
    #         "requests_left": self.requests_to_schedule_per_agent[agent_index][self.current_period]- self.current_app[agent_index],
    #         "agent_active": 1 if self.current_app[agent_index] < self.requests_to_schedule_per_agent[agent_index][self.current_period] else 0,
    #         "current_allocation": self.current_app_allocation[agent_index]
    #     }
    #     return observations
    
    def assign_task(self, agent_id, chosen_node):
        # print(agent_id, self.current_ms)
        # print(chosen_node, self.current_ms[agent_id])

        task_ordering = list(nx.topological_sort(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1]))
        current_task = task_ordering[self.current_ms[agent_id]]
        ms_cpu = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes[current_task]['cpu_request']
        
        if chosen_node < self.num_nodes_domain:
                node_costs = self.node_costs_domain[agent_id]
                power_consumption = self.power_consumption_domain[agent_id]
                node_id = chosen_node
        else:
            node_id = chosen_node-self.num_nodes_domain
            node_costs = self.node_costs_shared
            power_consumption = self.power_consumption_shared
        ms_cpu = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes[current_task]['cpu_request']
        if np.isnan(ms_cpu):
            ms_cpu = 0.1
        data_size = abs(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes[current_task]['computation_time'])
        parent_ms = list(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].pred[current_task].keys()) # Get the parent task names
        if len(parent_ms) == 0:
            if chosen_node < self.num_nodes_domain:
                wait_time = self.latency_from_user[agent_id]
            else:
                wait_time = self.latency_from_user[-1]
        else:
            wait_time = max([self.current_app_endtimes[agent_id][ms]+self.latencies[agent_id][self.current_app_allocation[agent_id][task_ordering.index(ms)]][chosen_node]-self.current_period for ms in parent_ms])
        computation_time = int(self.compute_time_required(chosen_node, agent_id, ms_cpu, data_size))
        self.app_spans[agent_id] += computation_time + wait_time

        valid = False
        scheduled_period = self.current_period
        while not valid:
            utilizations = deque([])
            for period in range(computation_time):
                if scheduled_period+wait_time+period >= 5000:
                    valid = False
                    break
                if chosen_node<self.num_nodes_domain:
                    if self.remaining_node_capacities_domain[agent_id][chosen_node][scheduled_period+period] < ms_cpu:
                        scheduled_period += 1
                        continue
                    utilizations.append(self.remaining_node_capacities_domain[agent_id][chosen_node][scheduled_period+period]-ms_cpu)
                else:
                    if self.remaining_node_capacities_shared[chosen_node-self.num_nodes_domain][scheduled_period+period] < ms_cpu:
                        scheduled_period += 1
                        continue
                    utilizations.append(self.remaining_node_capacities_shared[chosen_node-self.num_nodes_domain][scheduled_period+period]-ms_cpu)
            valid = True

        # valid = True
        # utilizations = deque([])
        # for period in range(computation_time):
        #     if self.current_period+wait_time+period >= 5000:
        #         valid = False
        #         break
        #     if chosen_node<self.num_nodes_domain:
        #         if self.remaining_node_capacities_domain[agent_id][chosen_node][self.current_period+period] < ms_cpu:
        #             valid = False
        #             break
        #         utilizations.append(self.remaining_node_capacities_domain[agent_id][chosen_node][self.current_period+period]-ms_cpu)
        #     else:
        #         if self.remaining_node_capacities_shared[chosen_node-self.num_nodes_domain][self.current_period+period] < ms_cpu:
        #             valid = False
        #             break
        #         utilizations.append(self.remaining_node_capacities_shared[chosen_node-self.num_nodes_domain][self.current_period+period]-ms_cpu)

        if valid:
            for period in range(computation_time):
                if chosen_node < self.num_nodes_domain:
                    self.allocation_per_timeslot_domain[agent_id][chosen_node, scheduled_period+wait_time+period].add((agent_id, self.current_app_total, self.current_ms[agent_id]))
                    self.remaining_node_capacities_domain[agent_id][chosen_node][scheduled_period+wait_time+period] -= ms_cpu
                else:
                    self.allocation_per_timeslot_shared[chosen_node-self.num_nodes_domain, scheduled_period+wait_time+period].add((agent_id, self.current_app_total, self.current_ms[agent_id]))
                    self.remaining_node_capacities_shared[chosen_node-self.num_nodes_domain][scheduled_period+wait_time+period] -= ms_cpu
            self.current_app_endtimes[agent_id][current_task] = scheduled_period + wait_time + computation_time
            self.current_app_allocation[agent_id][self.current_ms[agent_id]] = chosen_node
            self.operating_costs.append(node_costs[node_id]*computation_time)
            self.power_consumptions.append(power_consumption[node_id]*computation_time*ms_cpu)
            self.current_ms[agent_id] = self.current_ms[agent_id] + 1

            if chosen_node < self.num_nodes_domain:
                self.stored_in_edge += 1
            else:
                self.stored_in_cloud += 1
                if scheduled_period != self.current_period:
                    self.congestion_occurences += 1

            self.total_ms += 1
            if max(self.current_app_endtimes[agent_id].values()) == self.current_app_endtimes[agent_id][current_task]:
                # reward = -self.weights['power']*power_consumption[node_id]*computation_time*ms_cpu/self.max_consumption/self.max_latency-self.weights['latency']*(max(self.current_app_endtimes[agent_id])-self.current_period)/self.max_latency-self.weights['cost']*node_costs[node_id]*computation_time/self.max_cost/self.max_latency
                reward = -self.weights['latency']*(max(self.current_app_endtimes[agent_id].values())-self.current_period)/self.max_latency-self.weights['cost']*node_costs[node_id]*computation_time/self.max_cost/self.max_latency

            else:
                reward = -self.weights['cost']*node_costs[node_id]*computation_time/self.max_cost/self.max_latency
        else:
            reward = -500
            print("Episode finished due to mismanagement!")

        if self.app_is_allocated(agent_id):
            G = self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1]
            task_ordering = list(nx.topological_sort(G))
            allocation = self.current_app_allocation[agent_id]

            for idx, task in enumerate(task_ordering):
                parents = list(G.predecessors(task))
                for parent in parents:
                    parent_idx = task_ordering.index(parent)
                    if allocation[idx] != -1 and allocation[idx] == allocation[parent_idx]:
                        self.collocated_tasks += 1

            total_app_delay = max([self.current_app_endtimes[agent_id][ms]-self.current_period for ms in self.current_app_endtimes[agent_id].keys()])
            self.parallelism_ratio.append(self.app_spans[agent_id]/total_app_delay)
            self.app_spans[agent_id] = 0
            self.app_total_comp_times.append(total_app_delay)
            self.current_app_total += 1
            self.current_ms[agent_id] = 0
            self.current_app[agent_id] += 1
            self.current_app_allocation[agent_id] = (-1)*np.ones((self.max_tasks), dtype=np.int32)


        return reward, valid

    def step(self, actions):
        # print(self.current_period, self.current_app, self.current_ms)
        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        for agent_id, action in actions.items():
            agent_idx = int(agent_id.split('_')[-1])

            if action != self.no_op_action and self.requests_to_schedule_per_agent[agent_idx][self.current_period] - self.current_app[agent_idx] != 0:
                # Process the chosen action: assign current task to chosen node
                reward, validity = self.assign_task(agent_idx, action)
                if not validity and action >= self.num_nodes_domain:
                    self.congestion_occurences += 1
                    reward, validity = self.assign_task(agent_idx, random.choice(range(self.num_nodes_domain)))
                    reward -= 1
            else:
                validity = True
                reward = 0 
            rewards[agent_id] = reward
            terminateds[agent_id] = not validity  # Not terminating agents mid-period explicitly
            truncateds[agent_id] = False
            # infos[agent_id] = {}

        if all([self.period_is_scheduled(agent_id) for agent_id in range(self.nr_agents)]):
            # Check if the entire period is finished
            # print("entered")
            self.current_period = self.current_period + 1
            self.current_app = np.zeros(self.nr_agents, dtype=np.int32)
            self.current_ms = np.zeros(self.nr_agents, dtype=np.int32)
            self.current_app_endtimes = [{} for agent in range(self.nr_agents)]
            self.current_app_allocation = (-1)*np.ones((self.nr_agents, self.max_tasks), dtype=np.int32)

        if self.end_condition():
            # Check if the entire episode is finished
            # print("terminated")
            terminateds["__all__"] = True

        observations = self.get_all_observations()

        return observations, rewards, terminateds, truncateds, infos