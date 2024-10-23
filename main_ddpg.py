import cv2
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import os

# 定义超参数
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.001
LR_CRITIC = 0.002
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
UPDATE_EVERY = 20


def video_writer(frames, output_file, fps):
    height, width, _ = frames[0].shape
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用XVID编码
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    # 将每帧写入视频
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.write(frame)
    # 释放视频写入对象
    video_writer.release()
    print(f'Video has been saved as {output_file}')


# 定义 Actor 网络
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 确保输出在[-1, 1]之间


# 定义 Critic 网络
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size, log_dir):
        self.actor_local = Actor(state_size, action_size)
        self.critic_local = Critic(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.memory = deque(maxlen=BUFFER_SIZE)
        self.timestep = 0

        # 初始化目标网络
        self.update_targets(tau=1.0)

        # 初始化 TensorBoard Writer
        self.writer = SummaryWriter(log_dir)

    def update_targets(self, tau=TAU):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()
        return action

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, target_actions)
            target_Q = rewards + (GAMMA * target_Q * (1 - dones))

        # 更新 Critic
        Q_expected = self.critic_local(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        self.update_targets()

        # 记录损失
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.timestep)
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.timestep)
        self.timestep += 1

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


# 主程序
if __name__ == "__main__":
    env = gym.make('Pendulum-v1', g=1, render_mode="human")
    n_episodes = 500
    # 创建日志目录
    log_dir = f"runs/DDPN_Pendulum_custom_reward_{n_episodes}/"
    os.makedirs(log_dir, exist_ok=True)

    agent = DDPGAgent(state_size=3, action_size=1, log_dir=log_dir)

    model_path = f"models/DDPN_Pendulum_custom_reward_{n_episodes}.pth"

    if os.path.exists(model_path):
        print("加载已训练的模型...")
        agent.load_model(model_path)
    else:
        print("训练新的模型...")

        for episode in range(n_episodes):
            state = env.reset()[0]
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # 计算当前摆的角度
                angle = np.arctan2(state[1], state[0])  # 根据状态计算角度
                reward = 1 - abs(angle)  # 计算奖励
                agent.add_experience(state, action, reward, next_state, done)
                agent.learn()

                state = next_state
                total_reward += reward

            # 记录总奖励
            agent.writer.add_scalar('Reward/Total', total_reward, episode)
            print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}')

        # 保存训练好的模型
        agent.save_model(model_path)

    # 测试模型
    env = gym.make('Pendulum-v1', g=1, render_mode="rgb_array_list")
    for episode in range(3):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                frames = env.render()
                os.makedirs(f'./videos/ddpg_model_{n_episodes}', exist_ok=True)
                output_file = f'./videos/ddpg_model_{n_episodes}/test{episode}.mp4'
                video_writer(frames, output_file, 10)
            state = next_state
            angle = np.arctan2(state[1], state[0])
            reward = 1 - abs(angle)  # 计算奖励
            total_reward += reward

        print(f'Total Reward during Testing: {total_reward:.2f}')

    # 关闭 TensorBoard Writer
    agent.writer.close()
    env.close()
