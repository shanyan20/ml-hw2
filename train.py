import os

import torch
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter


# Custom Reward Function
def custom_reward(state, action):
    angle = np.arctan2(state[1], state[0])  # Calculate angle from state
    return 1.0 - abs(angle)  # Encourage the pendulum to stand upright


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


def train(agent, env, episodes, batch_size, target_update_freq, model_save_path):
    writer = SummaryWriter(f"./runs/DQN_Pendulum_custom_reward_{episodes}")

    for episode in range(episodes):
        random_state = np.array([np.random.uniform(-1, 1),  # cos(theta)
                                 np.random.uniform(-1, 1),  # sin(theta)
                                 np.random.uniform(-8, 8)])  # angular velocity
        state, _ = env.reset(options={"init_state": random_state})
        episode_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            torque = np.linspace(-agent.max_torque, agent.max_torque, agent.action_dim)[action_idx]
            next_state, _, terminated, truncated, _ = env.step([torque])
            done = terminated or truncated

            reward = custom_reward(state, [torque])
            agent.replay_buffer.add(state, action_idx, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = agent.train(batch_size)

            if done:
                print(f"Episode {episode + 1}, Reward: {episode_reward}")
                writer.add_scalar("Reward", episode_reward, episode)
                if loss is not None:
                    writer.add_scalar("Loss", loss, episode)
                if (episode + 1) % target_update_freq == 0:
                    agent.update_target_network()

    # Save the model after training
    torch.save(agent.q_network.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    writer.close()


def evaluate(agent, env, episodes, train_episodes):
    for episode in range(episodes):
        random_state = np.array([np.random.uniform(-1, 1),  # cos(theta)
                                 np.random.uniform(-1, 1),  # sin(theta)
                                 np.random.uniform(-8, 8)])  # angular velocity
        state, _ = env.reset(options={"init_state": random_state})
        done = False
        episode_reward = 0

        while not done:
            # 使用贪婪策略选择动作
            action_idx = agent.select_greedy_action(state)
            torque = np.linspace(-agent.max_torque, agent.max_torque, agent.action_dim)[action_idx]
            next_state, reward, terminated, truncated, _ = env.step([torque])
            done = terminated or truncated
            if done:
                frames = env.render()
                os.makedirs(f'./videos/dqn_model_{train_episodes}', exist_ok=True)
                output_file = f'./videos/dqn_model_{train_episodes}/test{episode}.mp4'
                video_writer(frames, output_file, 10)

            episode_reward += reward
            state = next_state
