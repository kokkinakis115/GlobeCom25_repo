import os

base_path = os.path.dirname(os.path.abspath(__file__))

import torch
import json
from dqn_train import DQN, flatten_obs
from env_v2 import Environment 


def test_dqn(avg_tasks=None, arrival_rate=None, agents=2):

    num_agents = agents
    test_iterations = 20
    # Load config
    config_path = os.path.join(base_path, "dqn_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Init model
    policy_net = DQN(cfg["input_dim"], cfg["action_dim"])
    policy_net.load_state_dict(torch.load(os.path.join(base_path, "./dqn_policy_1_agents_large_workload.pth")))
    policy_net.eval()

    env_config = {
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
        env_config["arrival_rate"] = arrival_rate

    env = Environment(params=env_config, avg_tasks=avg_tasks)
    print("Using subtrace: ", env.trace_path)
    print("Using arrival rate: ", env.request_arrival_rate)


    resulting_environments = []
    for i in range(test_iterations):
        # Create environment instance.
        env = Environment(params=env_config, avg_tasks=avg_tasks)

        obs_dict, _ = env.reset()
        states = {agent: flatten_obs(obs) for agent, obs in obs_dict.items()}
        done = False
        total_reward = 0

        while not done:
            actions = {}
            for agent in env.agents:
                if obs_dict[agent]["agent_active"] == 0:
                    actions[agent] = env.no_op_action
                else:
                    state_tensor = torch.FloatTensor(states[agent]).unsqueeze(0)
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                    action = int(torch.argmax(q_values, dim=1).item())
                    actions[agent] = action

            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            states = {agent: flatten_obs(obs) for agent, obs in obs_dict.items()}
            total_reward += sum(rewards.values())
            done = terminateds.get("__all__", False)
            for agent_id in range(env_config["agents"]):
                if rewards[f"agent_{agent_id}"] <= -500:
                    done = True
        
        print(f"Inference for Episode {i}. Total reward: ", total_reward)
        resulting_environments.append(env)

    return resulting_environments

if __name__ == "__main__":
    test_dqn()