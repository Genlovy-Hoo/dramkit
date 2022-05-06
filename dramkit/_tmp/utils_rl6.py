# -*- coding: utf-8 -*-

import time
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SARSA_Agent(object):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.01):
        self.gamma = gamma
        self.lr = lr
        self.__epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    @property
    def epsilon(self):
        return self.__epsilon

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax() # 选择当前最优动作
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done, next_action):
        # 动作价值估计
        u = reward + self.gamma * self.q[next_state, next_action] * (1 - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.lr * td_error


def play_sarsa(env, agent, train=False, render=False):
    '''SARS一个回合'''
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.learn(observation, action, reward, next_observation, done,
                        next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


class ExceptedSARSA_agent(object):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.01):
        self.gamma = gamma
        self.lr = lr
        self.__epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    @property
    def epsilon(self):
        return self.__epsilon

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        # 状态价值估计
        v = self.q[next_state].mean() * self.epsilon + \
            self.q[next_state].max() * (1 - self.epsilon)
        u = reward + self.gamma * v * (1 - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.lr * td_error


def play_qlearning(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward


class QLearning_agent(object):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.01):
        self.gamma = gamma
        self.lr = lr
        self.__epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    @property
    def epsilon(self):
        return self.__epsilon

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        u = reward + self.gamma * self.q[next_state].max() * (1 - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.lr * td_error


class DoubleQLearning_agent(object):
    def __init__(self, env, gamma=0.9, lr=0.1, epsilon=0.01):
        self.gamma = gamma
        self.lr = lr
        self.__epsilon = epsilon
        self.action_n = env.action_space.n
        self.q0 = np.zeros((env.observation_space.n, env.action_space.n))
        self.q1 = np.zeros((env.observation_space.n, env.action_space.n))

    @property
    def epsilon(self):
        return self.__epsilon

    def set_epsilon(self, epsilon):
        self.__epsilon = epsilon

    def decide(self, state):
        if np.random.uniform() > self.epsilon:
            action = (self.q0 + self.q1)[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action

    def learn(self, state, action, reward, next_state, done):
        if np.random.randint(2):
            self.q0, self.q1 = self.q1, self.q0
        a = self.q0[next_state].argmax()
        u = reward + self.gamma * self.q1[next_state, a] * (1. - done)
        td_error = u - self.q0[state, action]
        self.q0[state, action] += self.lr * td_error


class SARSALambda_Agent(SARSA_Agent):
    def __init__(self, env, lambd=0.9, beta=1.0,
                 gamma=0.9, lr=0.1, epsilon=0.01):
        super().__init__(env, gamma=gamma, lr=lr, epsilon=epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self, state, action, reward, next_state, done, next_action):
        # 更新资格迹
        self.e *= self.lambd * self.gamma
        self.e[state, action] = 1. + self.beta * self.e[state, action]

        # 更新价值
        u = reward + self.gamma * \
                self.q[next_state, next_action] * (1. - done)
        td_error = u - self.q[state, action]
        self.q += self.lr * self.e * td_error
        if done:
            self.e *= 0.


if __name__ == '__main__':
    strt_tm = time.time()

    # Taxi
    env = gym.make('Taxi-v3')
    # 进行一步
    state = env.reset()
    taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
    env.render()
    env.step(1)
    env.render()

    # # SARSA学习
    # agent = SARSA_Agent(env)
    # N = 10000
    # rewards = []
    # for k in range(N):
    #     rewards.append(play_sarsa(env, agent, train=True))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards)
    # plt.show()
    # print('Mean rewards of SARSA {} rounds: {}.'.format(N, np.mean(rewards)))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards[-200:])
    # plt.show()
    # # 使用学习好的agent
    # agent.set_epsilon(0.0)
    # rewards_ = [play_sarsa(env, agent) for _ in range(100)]
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards_)
    # plt.show()
    # print('Mean rewards of trained agent: {}.'.format(np.mean(rewards_)))

    # # 期望SARSA学习
    # agent = ExceptedSARSA_agent(env)
    # N = 5000
    # rewards = []
    # for k in range(N):
    #     rewards.append(play_qlearning(env, agent, train=True))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards)
    # plt.show()
    # print('Mean rewards of Exceptd-SARSA {} rounds: {}.'.format(N, np.mean(rewards)))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards[-200:])
    # plt.show()
    # # 使用学习好的agent
    # agent.set_epsilon(0.0)
    # rewards_ = [play_qlearning(env, agent) for _ in range(100)]
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards_)
    # plt.show()
    # print('Mean rewards of trained agent: {}.'.format(np.mean(rewards_)))

    # # Q学习
    # agent = QLearning_agent(env)
    # N = 5000
    # rewards = []
    # for k in range(N):
    #     rewards.append(play_qlearning(env, agent, train=True))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards)
    # plt.show()
    # print('Mean rewards of QLearning {} rounds: {}.'.format(N, np.mean(rewards)))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards[-200:])
    # plt.show()
    # # 使用学习好的agent
    # agent.set_epsilon(0.0)
    # rewards_ = [play_qlearning(env, agent) for _ in range(100)]
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards_)
    # plt.show()
    # print('Mean rewards of trained agent: {}.'.format(np.mean(rewards_)))

    # # 双重Q学习
    # agent = DoubleQLearning_agent(env)
    # N = 10000
    # rewards = []
    # for k in range(N):
    #     rewards.append(play_qlearning(env, agent, train=True))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards)
    # plt.show()
    # print('Mean rewards of DoubleQLearning {} rounds: {}.'.format(N, np.mean(rewards)))
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards[-200:])
    # plt.show()
    # # 使用学习好的agent
    # agent.set_epsilon(0.0)
    # rewards_ = [play_qlearning(env, agent) for _ in range(100)]
    # plt.figure(figsize=(12, 7))
    # plt.plot(rewards_)
    # plt.show()
    # print('Mean rewards of trained agent: {}.'.format(np.mean(rewards_)))

    # SARSA(lambda)学习 资格迹学习
    agent = SARSALambda_Agent(env)
    N = 10000
    rewards = []
    for k in range(N):
        rewards.append(play_sarsa(env, agent, train=True))
    plt.figure(figsize=(12, 7))
    plt.plot(rewards)
    plt.show()
    print('Mean rewards of SARSALambda {} rounds: {}.'.format(N, np.mean(rewards)))
    plt.figure(figsize=(12, 7))
    plt.plot(rewards[-200:])
    plt.show()
    # 使用学习好的agent
    agent.set_epsilon(0.0)
    rewards_ = [play_sarsa(env, agent) for _ in range(100)]
    plt.figure(figsize=(12, 7))
    plt.plot(rewards_)
    plt.show()
    print('Mean rewards of trained agent: {}.'.format(np.mean(rewards_)))


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
