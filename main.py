import os
import time
from datetime import datetime
import numpy as np
import torch

# import utils
import preprocessing_representations
from PPO import PPO_MARL
from env_v2 import Environment
# from train import train_with_curriculum

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # Initialize and train PPO
    print("============================================================================================")
    
    ####### initialize environment hyperparameters ######
    env_name = "CNA_Environment"
    
    has_continuous_action_space = False  # continuous action space; else discrete
    
    max_ep_len = 400                # max timesteps in one episode
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps
    
    print_freq = max_ep_len * 25       # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2.5           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################
    
    ## Note : print/log frequencies should be > than max_ep_len
    
    ################ PPO hyperparameters ################
    # update_timestep = int(max_ep_len // 2.5)      # update policy every n timesteps
    update_timestep = max_ep_len*10
    
    K_epochs = 80               # update policy for K epochs in one PPO update
    
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr_lstm = 0.001
    lr_gnn = 0.0005
    lr_actor = 0.0002       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    
    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################
    
    print("Training environment name : " + env_name)
    
    ################### Topology and Environment Hyperparameters ##################
    num_agents = 3
    params = {
        'time_periods': 20,
        'num_agents': num_agents,
        'num_nodes_domain': 5,
        'num_nodes_shared': 3,
        'capacity_range_domain': (10,40),
        'capacity_range_shared': (100, 200),
        'num_microservices': 10,
        'arrival_rate': 4,
        'look_ahead_window': 200,
        'weights': {'utilization':1 , 'cost': 1, 'latency': 1},
        'max_ms': 4
        }
    
    # env = DistributedStorageEnv(num_of_nodes, max_capacity, max_latency, fragmentation_schemes, num_of_requests, max_size, max_fragments, weights)
    env = Environment(params=params)
    
    # state space dimension
    # state_dim = utils.flatten_state(env.reset()).shape[0]
    
    # action space dimension
    
    action_dim = env.action_space.n
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
    run_num = 0
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
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################
    
    print("============================================================================================")
    
    ################# training procedure ################
    
    # initialize a PPO agent
    ppo_agent = PPO_MARL(params['num_microservices'], params['max_ms'], params['num_nodes_domain'], params['num_nodes_shared'], action_dim, lr_gnn, lr_lstm, lr_actor, lr_critic, gamma, K_epochs, eps_clip, params['num_agents'])
    
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
    # options_time_step = 0
    # actions_time_step = 0
    i_episode = 0

    # Initialize active nodes and requests
    # active_nodes = 10
    # active_requests = 20
    # increment_interval = 200  # Increase complexity every 200 episodes
    # randomization_start = 1000  # Start randomizing after 2000 episodes

    # Training loop
    while time_step <= max_training_timesteps:

        env.reset()
        done = False
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            full_state = env.get_joint_observation_space()
            # agent_observation_spaces = []
            agent_actions = []
            num_active_agents = 0
            active_agents = []
            for agent_id in range(num_agents):
                agent_state = env.get_observation_space(agent_id)
                if len(agent_state['requests']) > env.current_app:
                    # print(f"Taking action {t} for agent {agent_id}!")
                    active_agents.append(agent_id)
                    num_active_agents += 1

                    action_mask = env.get_action_mask(agent_id)
                    action = ppo_agent.select_action(agent_state, full_state, action_mask, agent_id, env.current_period)
                    agent_actions.append(action)
                else:
                    agent_actions.append(-1)
            _, rewards, is_terminals, _ = env.step(agent_actions)


            total_norm_reward = num_agents*sum(rewards)/num_active_agents
            current_ep_reward += total_norm_reward

            for agent_id in active_agents:
                # Saving reward and is_terminals
                ppo_agent.buffers[agent_id].rewards.append(total_norm_reward)
                ppo_agent.buffers[agent_id].is_terminals.append(is_terminals[agent_id])
                # ppo_agent.buffers[agent_id].total_rewards.append(total_norm_reward)

            if env.current_period == env.time_periods-1:
                done = True

            if env.app_is_allocated():
                env.reset_application()
            if all([env.period_is_scheduled(agent_id) for agent_id in range(num_agents)]):
                # print("Reseting Period!")
                env.reset_period()
                
                
            time_step += 1
            
            # update PPO agent
            if time_step % update_timestep == 0:
                print("Started Updating")
                ppo_agent.update()
                print("Stopped Updating")
            
            
            # log in logging file
            if time_step % log_freq == 0:
    
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
    
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
            if done or any(is_terminals):
                break
    
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