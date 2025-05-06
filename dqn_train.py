import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from env_v2 import Environment
import json
import matplotlib.pyplot as plt

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(rewards, window=10):
    # with open(path, "r") as f:
    #     rewards = json.load(f)

    episodes = list(range(len(rewards)))
    smoothed = moving_average(rewards, window)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Raw Rewards')
    plt.plot(episodes[window - 1:], smoothed, label=f'Smoothed (window={window})', color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

def flatten_obs(obs_dict):
    flat_list = []
    for key, value in obs_dict.items():
        if key == "node_capacities":
            capacities = []
            for node_capacity in value:
                capacities.append(min(node_capacity))
            flat_list.extend(capacities)
        elif isinstance(value, np.ndarray):
            flat_list.extend(value.flatten())
        else:
            flat_list.append(value)
    return np.array(flat_list, dtype=np.float32)

# DQN network definition using PyTorch.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512, 512]):
        super(DQN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Replay Buffer for experience replay.
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
def dqn_train():
    # Hyperparameters
    num_episodes = 1000
    batch_size = 4096
    gamma = 0.99
    lr = 1e-3
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 300  # episodes until epsilon decays
    target_update_freq = 10  # episodes
    buffer_capacity = 20000
    update_every = 2048
    step_count = 0
    num_updates_per_step = 10 # epochs
    ep_step_count_limit = 2048

    # Environment configuration
    env_config = {
        "time_periods": 11,
        "agents": 2,
        "num_nodes_domain": 10,
        "num_nodes_shared": 5,
        "capacity_range_domain": (40, 30),
        "capacity_range_shared": (80, 200),
        # "num_microservices": 10,
        "arrival_rate": 7,
        "look_ahead_window": 500,
        # "max_ms": 5,
        "window": 20,
        "max_tasks": 60,
        "task_features": 2,
        "max_dependencies": 100,
        "from_trace": True
    }

    # Create environment instance.
    env = Environment(env_config)

    # Use one agent's observation space to determine the input dimension.
    dummy_obs = flatten_obs(env.get_all_observations()["agent_0"])
    input_dim = dummy_obs.shape[0]
    # Action space size (same for all agents).
    action_dim = env.action_spaces["agent_0"].n-1

    # Create networks.
    policy_net = DQN(input_dim, action_dim)
    target_net = DQN(input_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Epsilon decay function.
    def get_epsilon(episode):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * episode / epsilon_decay)

    episode_rewards = []
    # Main training loop.
    for episode in range(num_episodes):
        obs_dict, _ = env.reset()
        # For each agent, flatten the observation.
        states = {agent: flatten_obs(obs) for agent, obs in obs_dict.items()}
        done = False
        episode_reward = {agent: 0 for agent in env.agents}
        ep_step_count = 0

        while not done:
            actions = {}
            # For each agent, select an action using epsilon-greedy.
            epsilon = get_epsilon(episode)
            for agent in env.agents:
                if obs_dict[agent]["agent_active"] == 0:
                    action = env.no_op_action
                else:
                    if random.random() < epsilon:
                        action = random.randint(0, action_dim - 1)
                    else:
                        state_tensor = torch.FloatTensor(states[agent]).unsqueeze(0)
                        q_values = policy_net(state_tensor)
                        action = int(torch.argmax(q_values, dim=1).item())
                actions[agent] = action

            # Step in the environment.
            next_obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            next_states = {agent: flatten_obs(obs) for agent, obs in next_obs_dict.items()}
            
            # Here we assume an episode is done when "__all__" is set.
            done = terminateds.get("__all__", False)
            # if done:
                # print("Episode finished.")
            
            # Store each agent's transition.
            for agent in env.agents:
                if actions[agent] == env.no_op_action:
                    continue  # Skip storing this transition
                replay_buffer.push(states[agent], actions[agent], rewards[agent], next_states[agent], terminateds[agent])
                episode_reward[agent] += rewards[agent]
                if rewards[agent] == -200:
                    done = True
            
            states = next_states
            
            step_count += 1
            ep_step_count += 1
            if len(replay_buffer) >= batch_size and step_count % update_every == 0:
                print("Training step...")
                for _ in range(num_updates_per_step):
                    states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
                    states_b = torch.FloatTensor(states_b)
                    actions_b = torch.LongTensor(actions_b).unsqueeze(1)
                    rewards_b = torch.FloatTensor(rewards_b)
                    next_states_b = torch.FloatTensor(next_states_b)
                    dones_b = torch.FloatTensor(dones_b)
                    
                    # Compute Q(s,a) using the policy network.
                    q_values = policy_net(states_b).gather(1, actions_b).squeeze(1)
                    # Compute the next Q values using the target network.
                    next_q_values = target_net(next_states_b).max(1)[0]
                    expected_q_values = rewards_b + gamma * next_q_values * (1 - dones_b)
                    
                    loss = nn.MSELoss()(q_values, expected_q_values.detach())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            if ep_step_count >= ep_step_count_limit:
                done = True
        
        # Update the target network periodically.
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        avg_reward = np.mean(list(episode_reward.values()))
        print(f"Episode {episode} - Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        episode_rewards.append(np.mean(list(episode_reward.values())))

    print("Training complete.")
    config = {"input_dim": int(input_dim), "action_dim": int(action_dim)}
    print(config)
    with open("dqn_config.json", "w") as f:
        json.dump(config, f)
    torch.save(policy_net.state_dict(), f"dqn_policy_{env_config["agents"]}_agents_large_workload.pth")

    # Plot results
    plot_results(episode_rewards, window=10)


if __name__ == "__main__":
    dqn_train()