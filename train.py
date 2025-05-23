import os
# import glob
import time
from datetime import datetime
import numpy as np
import torch

import utils
from PPO import PPO_MARL
from env_v2 import Environment

import matplotlib.pyplot as plt


################################### Training ###################################
def train():
    # Initialize and train PPO
    print("============================================================================================")
    
    ####### initialize environment hyperparameters ######
    num_agents = 3
    env_name = f"Environment_{num_agents}_agents_modified"
    max_ep_len = 2500
    # max_ep_len = 400                    # max timesteps in one episode
    max_training_timesteps = int(1.5e5)   # break training loop if timeteps > max_training_timesteps
    
    print_freq = max_ep_len * 5        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len               # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)
    
    #####################################################
    
    ## Note : print/log frequencies should be > than max_ep_len
    
    ################ PPO hyperparameters ################
    # update_timestep = int(max_ep_len // 2.5)      # update policy every n timesteps
    update_timestep = max_ep_len*5
    # update_timestep = max_ep_len*15
    # update_timestep = 600

    K_epochs = 30               # update policy for K epochs in one PPO update

    eps_clip = 0.12         # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_lstm = 0.0001 # lr_lstm = 0.000075
    lr_gnn = 0.0001 # lr_gnn = 0.000065

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################


    ################### Topology and Environment Hyperparameters ##################
    params = {
        "time_periods": 11,
        "agents": num_agents,
        "num_nodes_domain": 10,
        "num_nodes_shared": 10,
        "capacity_range_domain": (16, 24),
        "capacity_range_shared": (100, 200),
        "arrival_rate": 6,
        "look_ahead_window": 500,
        "window": 20,
        "max_tasks": 60,
        "task_features": 2,
        "max_dependencies": 100,
        "from_trace": True
    }

    print("Training environment name : " + env_name)
    env = Environment(params=params)

    # state space dimension
    # state_dim = utils.flatten_state(env.reset()).shape[0]

    # action space dimension
    action_dim = env.action_spaces["agent_0"].n-1

    # action_dim_master = tuple([action_dim[0], len(env.fragmentation_schemes)])
    # action_dim_worker = env.num_of_nodes
    # print("Master's action dimensions: ", action_dim_master)
    print("Action dimensions: ", action_dim)

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 1
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### Change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # checkpoint_path_master = directory + "PPO_Master_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path_worker = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    
    print("save checkpoint path : " + checkpoint_path)
    #####################################################
    
    
    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    # print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate lstm : ", lr_lstm)
    print("optimizer learning rate gnn : ", lr_gnn)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################
    
    print("============================================================================================")
    
    ################# training procedure ################
    
    ctde = True
    
    # initialize a PPO agent
    ppo_agent = PPO_MARL(params['num_nodes_domain'], params['num_nodes_shared'], action_dim, lr_gnn, gamma, K_epochs, eps_clip, params['agents'], inference=False, messages=False, modified=True)
    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    
    print("============================================================================================")
    
    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # Initialize active nodes and requests
    # active_nodes = 10
    # active_requests = 20
    # increment_interval = 200  # Increase complexity every 200 episodes
    # randomization_start = 1000  # Start randomizing after 2000 episodes

    # creating the first plot and frame
    plt.ion()  # turning interactive mode on
    plot_rewards = [0]
    plot_timesteps = [0]
    fig, ax = plt.subplots()
    graph = ax.plot(plot_timesteps, plot_rewards, color = 'g')[0]
    
    # Training loop
    while time_step <= max_training_timesteps:
        # if time_step % 100 == 0:
        # print(time_step, i_episode)
        reward = 0

        env.reset()
        done = False
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            
            # full_state = env.get_joint_observation_space()
            # agent_observation_spaces = []
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
            # print("Actions: ", agent_actions)
            # print("Rewards: ", rewards)
            
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
            current_ep_reward += sum(rewards)
            
            
            for i, agent_id in enumerate(active_agents):
                # Saving reward and is_terminals
                reward_for_normalization = total_norm_reward+rewards[i] if total_norm_reward+rewards[i] != 0 else 0.001
                ppo_agent.buffers[agent_id].rewards.append(0.9*rewards[i]/reward_for_normalization+0.1*total_norm_reward/reward_for_normalization)
                ppo_agent.buffers[agent_id].is_terminals.append(is_terminals[f"agent_{agent_id}"])
            if done:
                for agent_id in range(num_agents):
                    if len(ppo_agent.buffers[agent_id].is_terminals) != 0:
                        ppo_agent.buffers[agent_id].is_terminals[-1] = True #Ensure that last action in episode has done=True for all agents
                
            # if env.app_is_allocated():
            #     # print("Allocated all microservices!")
            #     # print(env.requests_to_schedule_per_agent[:,env.current_period], env.current_app)
            #     env.reset_application()
            # if all([env.period_is_scheduled(agent_id) for agent_id in range(num_agents)]):
            #     # print("Reseting Period!", [env.period_is_scheduled(agent_id) for agent_id in range(num_agents)])
            #     env.reset_period()
                
                
            time_step += 1
            
            # update PPO agent
            if time_step % update_timestep == 0:
                # print("Started Updating")
                ppo_agent.update()
                # print("Stopped Updating")
            
            
            # log in logging file
            if time_step % log_freq == 0:
    
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
    
                plot_rewards.append(log_avg_reward)
                plot_timesteps.append(time_step)
                # removing the older graph
                graph.remove()
                graph = plt.plot(plot_timesteps, plot_rewards, color = 'g')[0]
                plt.pause(0.1)
                
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
    
                log_running_reward = 0
                log_running_episodes = 0
    
            # printing average reward
            if time_step % print_freq == 0:
    
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
    
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
    
                print_running_reward = 0
                print_running_episodes = 0
    
            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("Savings models at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path_worker)
                print("Models saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
    
            # break; if the episode is over
            if done:
                # print(done, any(is_terminals))
                break
            # print(t)

        if i_episode==0:
            plot_rewards = [current_ep_reward]
            plot_timesteps = [time_step]
        
            fig, ax = plt.subplots()
            graph = ax.plot(time_step, reward,color = 'g')[0]
        
        print_running_reward += current_ep_reward
        print_running_episodes += 1
    
        log_running_reward += current_ep_reward
        log_running_episodes += 1
    
        i_episode += 1
    
        # Increment active nodes and requests gradually
        # if i_episode % increment_interval == 0 and i_episode < randomization_start:
        #     active_nodes = min(env.num_of_nodes, active_nodes+10)
        #     active_requests = min(env.num_of_requests, active_requests+10)
    
    log_f.close()
    env.close()
    
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    
    
if __name__ == '__main__':

    train()