# model.py
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import torch.optim as optim
import random


# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)


# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, max_torque, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_torque = max_torque
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(max_size=10000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)  # Random action
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()  # Action with highest Q-value

    def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return q_values.argmax().item()  # Action with highest Q-value

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]

        loss = nn.SmoothL1Loss()(q_value, next_q_value)  # Use SmoothL1Loss for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()  # Return the loss value for tracking

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
