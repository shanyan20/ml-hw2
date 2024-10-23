# main.py
import os
import gym
import torch
from model import DQNAgent
from train import train, evaluate


def main():
    # Load the environment
    env = gym.make('Pendulum-v1', g=1, render_mode="human")

    # Define hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = 10  # Number of discrete torque actions
    max_torque = env.max_torque
    episodes = 500
    batch_size = 128
    target_update_freq = 5
    model_save_path = f"./models/dqn_pendulum_model_{episodes}.pth"

    # Create the agent
    agent = DQNAgent(state_dim, action_dim, max_torque)

    # Check if the model already exists
    if os.path.exists(model_save_path):
        print("Loading existing model...")
        agent.q_network.load_state_dict(torch.load(model_save_path))
        agent.q_network.eval()  # Set the model to evaluation mode
        env = gym.make('Pendulum-v1', g=1, render_mode="rgb_array_list")
        evaluate(agent, env, 3, episodes)  # Evaluate the model
    else:
        print("Training new model...")
        train(agent, env, episodes, batch_size, target_update_freq, model_save_path)  # Train the model
        agent.q_network.eval()
        env = gym.make('Pendulum-v1', g=1, render_mode="rgb_array_list")
        evaluate(agent, env, 3, episodes)  # Evaluate the model
    env.close()


if __name__ == "__main__":
    main()
