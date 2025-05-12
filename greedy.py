import numpy as np
import utils
from collections import deque
from env_v2 import Environment
import networkx as nx


def select_action_greedy(env, agent_id):

    task_ordering = list(nx.topological_sort(env.requests_per_agent[agent_id][env.current_period][env.current_app[agent_id]][1]))
    current_task = task_ordering[env.current_ms[agent_id]]
    ms_cpu = env.requests_per_agent[agent_id][env.current_period][env.current_app[agent_id]][1].nodes[current_task]['cpu_request']

    # print(agent_id, self.current_app_endtimes, task_ordering)
    # print(self.requests_per_agent[agent_id][self.current_period][self.current_app[agent_id]][1].nodes)
    # print(self.current_app, self.current_ms)   

    #####

    num_nodes_domain = env.num_nodes_domain
    num_nodes_shared = env.num_nodes_shared
    valid_nodes = np.ones(num_nodes_domain+num_nodes_shared)
    if sum(valid_nodes) == 0:
        print('invalid')
    
    scores = []
    for node in range(num_nodes_domain+num_nodes_shared):
        if node < env.num_nodes_domain:
            node_costs = env.node_costs_domain[agent_id]
            power_consumption = env.power_consumption_domain[agent_id]
            node_id = node
        else:
            node_id = node-env.num_nodes_domain
            node_costs = env.node_costs_shared
            power_consumption = env.power_consumption_shared
        if np.isnan(ms_cpu):
            ms_cpu = 0.1
        data_size = abs(env.requests_per_agent[agent_id][env.current_period][env.current_app[agent_id]][1].nodes[current_task]['computation_time'])

        # print(data_size, ms_cpu)
        parent_ms = list(env.requests_per_agent[agent_id][env.current_period][env.current_app[agent_id]][1].pred[current_task].keys()) # Get the parent task names
    
        if node < env.num_nodes_domain:
            node_costs = env.node_costs_domain[agent_id]
            power_consumption = env.power_consumption_domain[agent_id]
            node_id = node
        else:
            node_id = node-env.num_nodes_domain
            power_consumption = env.power_consumption_shared
            node_costs = env.node_costs_shared
        if len(parent_ms) == 0:
            if node < env.num_nodes_domain:
                wait_time = env.latency_from_user[agent_id]
            else:
                wait_time = env.latency_from_user[-1]
        else:
            wait_time = max([env.current_app_endtimes[agent_id][ms]+env.latencies[agent_id][env.current_app_allocation[agent_id][task_ordering.index(ms)]][node]-env.current_period for ms in parent_ms])
        computation_time = int(env.compute_time_required(node, agent_id, ms_cpu, data_size))
        utilizations = deque([])

        for period in range(computation_time):
            if node<env.num_nodes_domain:
                if env.remaining_node_capacities_domain[agent_id][node][env.current_period+period] < ms_cpu:
                    break
                utilizations.append(env.remaining_node_capacities_domain[agent_id][node][env.current_period+period]-ms_cpu)
            else:
                if env.remaining_node_capacities_shared[node-env.num_nodes_domain][env.current_period+period] < ms_cpu:
                    break
                utilizations.append(env.remaining_node_capacities_shared[node-env.num_nodes_domain][env.current_period+period]-ms_cpu)
        if len(utilizations)==0:
            node_score = -10000
        else:
            if env.current_app_endtimes[agent_id]:
                if max(env.current_app_endtimes[agent_id].values()) == env.current_period + wait_time + computation_time:
                    # reward = -self.weights['power']*power_consumption[node_id]*computation_time*ms_cpu/self.max_consumption/self.max_latency-self.weights['latency']*(max(self.current_app_endtimes[agent_id])-self.current_period)/self.max_latency-self.weights['cost']*node_costs[node_id]*computation_time/self.max_cost/self.max_latency
                    node_score = -env.weights['latency']*(max(env.current_app_endtimes[agent_id].values())-env.current_period)/env.max_latency-env.weights['cost']*node_costs[node_id]*computation_time/env.max_cost/env.max_latency
                else:
                    node_score = -env.weights['cost']*node_costs[node_id]*computation_time/env.max_cost/env.max_latency
            else:
                node_score = -env.weights['cost']*node_costs[node_id]*computation_time/env.max_cost/env.max_latency

        scores.append(node_score)
    sorted_nodes = np.argsort(scores)
    chosen_node = -1
    for node in sorted_nodes[::-1]:
        if valid_nodes[node] == 1:
            chosen_node = node
            break
    if chosen_node == -1:
        chosen_node = 0 #underutilization
    return chosen_node


def test_greedy(avg_tasks=None, arrival_rate=None):
    print("============================================================================================")
    
    num_agents = 3
    env_name = f"CNA_Environment_{num_agents}_agents"
    
    max_ep_len = 5000           # max timesteps in one episode


    total_test_episodes = 10    # total num of testing episodes

    #####################################################

    params = {
        "time_periods": 11,
        "agents": num_agents,
        "num_nodes_domain": 10,
        "num_nodes_shared": 3,
        "capacity_range_domain": (20, 40),
        "capacity_range_shared": (60, 120),
        # "num_microservices": 10,
        "arrival_rate": 6,
        "look_ahead_window": 500,
        # "max_ms": 5,
        "window": 20,
        "max_tasks": 60,
        "task_features": 2,
        "max_dependencies": 100,
        "from_trace": True
    }
    if arrival_rate is not None:
        params["arrival_rate"] = arrival_rate
    
    print("Training environment name : " + env_name)
    env = Environment(params=params, avg_tasks=avg_tasks)
    print("Using subtrace: ", env.trace_path)
    print("Using arrival rate: ", env.request_arrival_rate)

    print("--------------------------------------------------------------------------------------------")
    
    test_running_reward = 0
    resulting_environments = []
    
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        env.reset()
        done = False
        for t in range(1, max_ep_len+1):
            # full_state = env.get_joint_observation_space()
            # agent_observation_spaces = []
            agent_actions = []
            num_active_agents = 0
            active_agents = []
            for agent_id in range(num_agents):
                agent_state = env.get_all_observations()['agent_{}'.format(agent_id)]
                if agent_state['requests_left'] >= 0 and agent_state['agent_active'] == 1:
                    active_agents.append(agent_id)
                    num_active_agents += 1

                    action = select_action_greedy(env, agent_id)
                    agent_actions.append(action)
                else:
                    agent_actions.append(env.no_op_action)

            action_dict = {}
            for agent_id in range(num_agents):
                action_dict["agent_{}".format(agent_id)] = agent_actions[agent_id]
            _, rewards_dict, is_terminals, _, _ = env.step(action_dict)

            rewards = [rewards_dict[f"agent_{agent_id}"] for agent_id in range(num_agents)]
            # done = is_terminals.get("__all__", False)
            for agent_id in range(num_agents):
                if is_terminals[f"agent_{agent_id}"]:
                    done = True

            if env.current_period == env.time_periods-1:
                done = True
                is_terminals["__all__"] = True
            
            total_norm_reward = num_agents*sum(rewards)/num_active_agents if num_active_agents != 0 else 0
            # print(total_norm_reward, agent_actions)
            ep_reward += sum(rewards)
    
            # break; if the episode is over
            if done:
                break

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
        resulting_environments.append(env)
        env.close()
        env = Environment(params=params)

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")
    
    return resulting_environments