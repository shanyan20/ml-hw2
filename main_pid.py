import os
import cv2
import gym
import math

# Env Construction
env = gym.make('Pendulum-v1', g=1, render_mode="rgb_array_list")

# Env Reset
observation, info = env.reset()
max_torque = 10
env.max_torque = max_torque
env.action_space = gym.spaces.Box(-max_torque, max_torque)

# Env Info
print("observation space: " + str(env.observation_space))
print("action space: " + str(env.action_space))
step = 0
episode = 0

# 初始化PID参数
P = 1.2
I = 0.0
D = 10.0

# 初始化变量
prev_error = 0
integral = 0


def get_current_state(obs):
    x = obs[0]
    y = obs[1]
    w = obs[2]
    theta = math.atan2(y, x)
    return theta, w


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


for i in range(3):
    observation, info = env.reset()
    prev_error = 0
    integral = 0
    while True:

        # 获取当前摆杆的角度和角速度
        current_angle, current_angular_velocity = get_current_state(observation)

        # 计算偏差
        error = 0 - current_angle
        # 计算积分误差
        integral = integral + error

        # 计算微分误差
        derivative = error - prev_error

        # 计算控制输出
        control_output = P * error + I * integral + D * derivative
        # 施加力矩控制倒立摆
        # apply_torque(control_output)
        action = [control_output]
        observation, reward, terminated, truncated, info = env.step(action)

        # 更新前一次的偏差
        prev_error = error

        if terminated or truncated:
            frames = env.render()
            os.makedirs(f'./videos/pid_i={I}/', exist_ok=True)
            output_file = f'./videos/pid_i={I}/test{i}.mp4'
            video_writer(frames, output_file, 10)
            break

env.close()
