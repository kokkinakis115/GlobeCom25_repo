import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO import PPO_MARL
from env_v2 import Environment
import numpy as np

#################################### Testing ###################################
def test_model(avg_tasks=None, arrival_rate=None, agents=2):
    ################## hyperparameters ##################
    
    # np.random.seed(1)
    num_agents = agents
    num_agents_trained = 3
    env_name = f"CNA_Environment_2_agents_large_workload_2"
    
    max_ep_len = 2000           # max timesteps in one episode


    total_test_episodes = 20    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr_lstm = 0.001
    lr_gnn = 0.0002
    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    params = {
        "time_periods": 11,
        "agents": num_agents,
        "num_nodes_domain": 10,
        "num_nodes_shared": 10,
        "capacity_range_domain": (20, 30),
        "capacity_range_shared": (100, 200),
        # "num_microservices": 10,
        "arrival_rate": 5,
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

    # state space dimension

    # action space dimension

    action_dim = env.action_spaces["agent_0"].n-1

    ctde = True
    
    # initialize a PPO agent
    ppo_agent = PPO_MARL(params['num_nodes_domain'], params['num_nodes_shared'], action_dim, lr_gnn, gamma, K_epochs, eps_clip, params['agents'], inference=True)
    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    
    test_running_reward = 0
    resulting_environments = []
    
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        env.reset()
        done = False
        for t in range(1, max_ep_len+1):
    
            agent_actions = []
            num_active_agents = 0
            active_agents = []
            for agent_id in range(num_agents):
                agent_state = env.get_all_observations()['agent_{}'.format(agent_id)]
                if agent_state['requests_left'] >= 0 and agent_state['agent_active'] == 1:
                    # print(f"Picking node {t} for agent {agent_id}!")
                    active_agents.append(agent_id)
                    num_active_agents += 1

                    action_mask = env.get_action_mask(agent_id)
                    # print(action_mask)
                    action = ppo_agent.select_action(agent_state, agent_id, action_mask)
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
                    # print("entered")
                    # print(agent_state)
                    done = True

            if env.current_period == env.time_periods-1:
                # print(num_active_agents)
                # print("entered")
                done = True
                is_terminals["__all__"] = True
            
            total_norm_reward = num_agents*sum(rewards)/num_active_agents if num_active_agents != 0 else 0
            # print(total_norm_reward, agent_actions)
            ep_reward += sum(rewards)
            # break; if the episode is over
            if done:
                break
        # clear buffer
        for agent_id in range(num_agents):
            ppo_agent.buffers[agent_id].clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
        resulting_environments.append(env)
        env.close()
        env = Environment(params=params, avg_tasks=avg_tasks)

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")
    
    return resulting_environments

# if __name__ == '__main__':

#     test()

