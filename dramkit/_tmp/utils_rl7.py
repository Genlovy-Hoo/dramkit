# -*- coding: utf-8 -*-

import time
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TileCoder:
    '''
    线性近似最优策略求解，砖瓦编码
    '''
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features: # 冲突处理
            return hash(codeword) % self.features
        self.codebook[codeword] = count
        return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) /
                    self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SARSAAgent:
    '''SARSA 算法'''
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n # 动作数
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - \
                env.observation_space.low # 观测空间范围
        self.encoder = TileCoder(layers, features) # 砖瓦编码器
        self.w = np.zeros(features) # 权重
        self.gamma = gamma # 折扣
        self.learning_rate = learning_rate # 学习率
        self.epsilon = epsilon # 探索

    def encode(self, observation, action): # 编码
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action): # 动作价值
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation): # 判决
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in
                  range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, next_observation, done,
              next_action): # 学习
        u = reward + (1. - done) * self.gamma * \
            self.get_q(next_observation, next_action)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)


def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation) # 终止状态时此步无意义
        if train:
            agent.learn(observation, action, reward, next_observation,
                        done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


if __name__ == '__main__':
    strt_tm = time.time()

    # MountainCar
    env = gym.make('MountainCar-v0')
    # 环境信息
    env.seed(0)
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('位置范围 = {}'.format((env.unwrapped.min_position,
                                 env.unwrapped.max_position)))
    print('速度范围 = {}'.format((-env.unwrapped.max_speed,
            env.unwrapped.max_speed)))
    print('目标位置 = {}'.format(env.unwrapped.goal_position))

    # 一直向右，不能成功
    positions, velocities = [], []
    observation = env.reset()
    while True:
        env.render()
        positions.append(observation[0])
        velocities.append(observation[1])
        next_observation, reward, done, _ = env.step(2)
        if done:
            break
        observation = next_observation
    time.sleep(2)
    env.close()

    if next_observation[0] > 0.5:
        print('成功到达')
    else:
        print('失败退出')

    # 绘制位置和速度图像
    fig, ax = plt.subplots()
    ax.plot(positions, label='position')
    ax.plot(velocities, label='velocity')
    ax.legend()
    plt.show()

    # SARSA
    agent = SARSAAgent(env)

    # 训练
    episodes = 400
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play_sarsa(env, agent, train=True)
        episode_rewards.append(episode_reward)
    plt.figure()
    plt.plot(episode_rewards)
    plt.show()

    # 测试
    agent.epsilon = 0. # 取消探索
    episode_rewards = [play_sarsa(env, agent) for _ in range(100)]
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
            len(episode_rewards), np.mean(episode_rewards)))


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
