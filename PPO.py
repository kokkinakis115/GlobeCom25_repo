import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from nn_preprocessing import GNNAttention, GNNCritic, GNNApplication
import preprocessing_representations
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence



################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.graph_states = []
        self.req_states = []
        self.requests_left = []
        self.seq_lengths = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.action_masks = []
        self.critic_states_graph = []
        self.critic_states_req = []
    
    def clear(self):
        del self.actions[:]
        del self.graph_states[:]
        del self.req_states[:]
        del self.requests_left[:]
        del self.seq_lengths[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.action_masks[:]
        del self.critic_states_graph[:]
        del self.critic_states_req[:]

class ActorCriticWorker(nn.Module):
    def __init__(self, num_nodes, num_agents, action_dims, inference=False):
        super(ActorCriticWorker, self).__init__()
        self.zeros_arr = torch.zeros(action_dims, dtype=torch.float32)
        self.action_dims = action_dims
        self.inference = inference
        self.num_agents = num_agents
        
        application_gnn_input_dim = 3 # CPU, Data, is_current
        application_embedding_dim = 3

        self.application_gnn = GNNApplication(input_dim=application_gnn_input_dim, hidden_dim_gnn=16, output_dim=application_embedding_dim) # 3 node features CPU, Data, is_current

        input_dim_gnn_infr = 4 # CPU, Cost, Coeff, has_predecessor, has_latest_request
        self.policy_gnn = GNNAttention(input_dim=input_dim_gnn_infr, hidden_dim_gnn=64, embedding_dim=application_embedding_dim+1, attention_dim=32, output_dim=1, num_nodes=num_nodes)

        self.critic_gnn = GNNCritic(input_dim=input_dim_gnn_infr, hidden_dim_gnn=64, embedding_dim=application_embedding_dim+1, attention_dim=32, output_dim=1, num_nodes=num_nodes)

    def forward(self):
        raise NotImplementedError


    def act(self, state, action_mask=None, batch=False):

        dependencies_repr = preprocessing_representations.produce_local_app_repr(state['request_features'], state['request_dependencies'], state['current_ms'], state['num_tasks'], state['num_dependencies'])        
        
        #Infrastructure embeddings
        # For the current request, we need to get the predecessors
        predecessors = []
        for i in range(state['request_dependencies'].shape[0]):
            if state['request_dependencies'][i][1] == 4:
                predecessors.append(int(state['request_dependencies'][i][0]))

        # For the current period we need the minimum available capacities
        cpu = [int(min(node_capacity)) for node_capacity in state['node_capacities']]

        graph_repr = preprocessing_representations.produce_infr_repr(cpu, state['node_costs'], state['device_coef'], state['latencies'], state['current_allocation'], predecessors)
        ### Tensorize Inputs
        dependencies_repr = Batch.from_data_list([dependencies_repr])
        graph_repr_batch = Batch.from_data_list([graph_repr])
        ###

        ### Actor Model
        req_embeddings = self.application_gnn(dependencies_repr).squeeze() # process current request with attention to future requests
        # print(state["requests_left"])
        # print("req_embeddings", req_embeddings.shape)

        requests_left = torch.tensor([state['requests_left']], dtype=torch.float32)
        # print("requests_left", requests_left.shape)
        req_embeddings = torch.cat((req_embeddings, requests_left), dim=0) # Concatenate along the feature dimension
        
        if not batch:
            req_embeddings = req_embeddings.unsqueeze(0)
        node_scores = self.policy_gnn(graph_repr_batch, req_embeddings).squeeze() # process graph with attention to requests

        # print("node_scores", node_scores)
        action_probs = torch.softmax(node_scores, dim=0) # Shape: (num_of_nodes,)
    
        if action_mask is not None:
            action_probs = action_probs * action_mask  # Zero out invalid actions
            if torch.equal(action_probs, self.zeros_arr):
                action_probs = torch.ones(self.action_dims)
            action_probs = action_probs / action_probs.sum()  # Re-normalize the probabilities

        dist = Categorical(action_probs)
    
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if not self.inference:  
            if not batch:
                req_embeddings = req_embeddings.unsqueeze(0)
            state_val = self.critic_gnn(graph_repr_batch, req_embeddings).squeeze()
        else:
            state_val = torch.zeros(1)

        return action.detach(), action_logprob.detach(), state_val.detach(), dependencies_repr, graph_repr, requests_left


    def evaluate(self, graph_state, dependencies_repr, action, requests_left, action_mask=None):
        
        ### Actor Model
        req_embeddings = self.application_gnn(dependencies_repr).squeeze() # process current request with attention to future requests
        requests_left = requests_left.unsqueeze(1)
        req_embeddings = torch.cat((req_embeddings, requests_left), dim=1) # Concatenate along the feature dimension
        node_embeddings = self.policy_gnn(graph_state, req_embeddings).squeeze() # process graph with attention to requests
        
        action_probs = torch.softmax(node_embeddings, dim=0) # Shape: (num_of_nodes,)
        
        if action_mask is not None:
            action_probs = action_probs * action_mask  # Zero out invalid actions
            for i, tnsr in enumerate(action_probs):
                if torch.equal(action_probs[i], self.zeros_arr):
                    action_probs[i] = torch.ones(self.action_dims)
            action_probs = action_probs / action_probs.sum()  # Normalize
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_values = self.critic_gnn(graph_state, req_embeddings).squeeze()
 
        return action_logprobs.T, state_values, dist_entropy.mean()
    

class PPO_MARL:
    def __init__(self, nodes_per_agent, shared_nodes, action_dims, lr_gnn, gamma, K_epochs, eps_clip, number_of_agents, inference=False):
        self.nodes_per_agent = nodes_per_agent
        self.action_dims = action_dims
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCriticWorker(nodes_per_agent+shared_nodes, number_of_agents, action_dims, inference).to(device)
        self.policy_old = ActorCriticWorker(nodes_per_agent+shared_nodes, number_of_agents, action_dims, inference).to(device)
        self.buffer = RolloutBuffer()
        self.num_agents = number_of_agents
        self.buffers = [RolloutBuffer() for _ in range(number_of_agents)]
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.policy_gnn.parameters(), 'lr': lr_gnn},
            {'params': self.policy.application_gnn.parameters(), 'lr': lr_gnn},
            {'params': self.policy.critic_gnn.parameters(), 'lr': lr_gnn},
        ])

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, obs_state, agent_id=0, action_mask=None):
        with torch.no_grad():
            action_mask = torch.FloatTensor(action_mask).to(device)
            # print("action_mask", action_mask)
            actions, action_logprobs, state_val, dependencies_repr, graph_repr, requests_left = self.policy_old.act(obs_state, action_mask)
        
        self.buffers[agent_id].action_masks.append(action_mask)
        self.buffers[agent_id].graph_states.append(graph_repr)
        self.buffers[agent_id].req_states.append(dependencies_repr)
        self.buffers[agent_id].requests_left.append(requests_left)
        self.buffers[agent_id].actions.append(actions)
        self.buffers[agent_id].logprobs.append(action_logprobs)
        self.buffers[agent_id].state_values.append(state_val)

        return actions.item()


    def update(self):
        # Monte Carlo estimate of returns
        all_rewards, all_actions, all_logprobs, all_graph_states, all_req_states, all_requests_left, all_state_values, all_action_masks = [], [], [], [], [], [], [], []

        # Traverse all agent buffers and gather experiences
        for agent_id in range(self.num_agents):
            rewards = []
            discounted_reward = 0
    
            # Monte Carlo estimate of returns for each agent
            for reward, is_terminal in zip(reversed(self.buffers[agent_id].rewards), reversed(self.buffers[agent_id].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
    
            # Normalize the rewards for each agent
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # Convert list to tensor and detach
            all_rewards += rewards
            all_graph_states += self.buffers[agent_id].graph_states
            all_req_states += self.buffers[agent_id].req_states
            all_requests_left += self.buffers[agent_id].requests_left
            all_actions += self.buffers[agent_id].actions
            all_action_masks += self.buffers[agent_id].action_masks
            all_logprobs += self.buffers[agent_id].logprobs
            all_state_values += self.buffers[agent_id].state_values


        # Stack all agent experiences along the batch dimension
        all_rewards = torch.squeeze(torch.stack(all_rewards, dim=0)).detach().to(device)
        all_graph_states = Batch.from_data_list(all_graph_states)
        all_req_states =  Batch.from_data_list(all_req_states)
        all_actions = torch.squeeze(torch.stack(all_actions, dim=0)).detach().to(device)
        all_logprobs = torch.squeeze(torch.stack(all_logprobs, dim=0)).detach().to(device)
        all_state_values = torch.squeeze(torch.stack(all_state_values, dim=0)).detach().to(device)
        all_requests_left = torch.squeeze(torch.stack(all_requests_left, dim=0)).detach().to(device)
        all_action_masks = torch.squeeze(torch.stack(all_action_masks, dim=0)).detach().to(device)
        
        # Calculate advantages
        advantages = all_rewards - all_state_values
        
        mini_batch_size = 512
        batch_size = all_rewards.size(0)
        
        for _ in range(self.K_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, all_rewards.size(0))

                # Slice the batch
                mini_rewards = all_rewards[start:end]
                mini_actions = all_actions[start:end]
                mini_logprobs = all_logprobs[start:end]
                mini_state_values = all_state_values[start:end]
                mini_action_masks = all_action_masks[start:end]
                
                # Graph states, req states, and req lstm should also be sliced
                mini_graph_states = Batch.from_data_list(all_graph_states[start:end])
                mini_req_states = Batch.from_data_list(all_req_states[start:end])
                mini_requests_left = all_requests_left[start:end]
        
                # Evaluate the mini-batch
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mini_graph_states, mini_req_states,
                    mini_actions, mini_requests_left,
                    mini_action_masks
                )
        
                # Match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
        
                # Calculate the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - mini_logprobs.detach())
        
                # Calculate Surrogate Loss
                advantages = mini_rewards - mini_state_values
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
                # Calculate final loss
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mini_rewards) - 0.1 * dist_entropy
                # print(-torch.min(surr1, surr2), torch.mean(0.5 * self.MseLoss(state_values, mini_rewards)), torch.mean(- 0.1 * dist_entropy))
                # print(torch.mean(loss))
                # Update policy
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        for agent_id in range(self.num_agents):
            self.buffers[agent_id].clear()
       

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
