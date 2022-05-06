# -*- coding: utf-8 -*-

import time
import gym
import numpy as np
import matplotlib.pyplot as plt


def play_policy(env, policy=None):
    '''Play 1 round'''
    total_reward = 0
    observation = env.reset()
    print('初始observation = {}'.format(observation))
    while True:
        print('玩家 = {}, 庄家 = {}'.format(env.player, env.dealer))
        action = policy()
        print('动作 = {}'.format(action))
        observation, reward, done, _ = env.step(action)
        print('观测 = {}, 奖励 = {}, 结束指示 = {}'.format(observation, reward, done))
        total_reward += reward
        if done:
            break
    return total_reward


def ob2state(observation):
    '''根据观测获取状态索引'''
    return observation[0], observation[1], int(observation[2])


def play_policy2(env, policy):
    '''Play 1 round'''
    total_reward = 0
    observation = env.reset()
    print('初始observation = {}'.format(observation))
    while True:
        print('玩家 = {}, 庄家 = {}'.format(env.player, env.dealer))
        state = ob2state(observation)
        action = np.random.choice(env.action_space.n, p=policy[state])
        print('动作 = {}'.format(action))
        observation, reward, done, _ = env.step(action)
        print('观测 = {}, 奖励 = {}, 结束指示 = {}'.format(observation, reward, done))
        total_reward += reward
        if done:
            break
    return total_reward


def evaluate_action_monte_carlo(env, policy, episode_num=100000):
    '''蒙特卡罗方法评估动作价值函数'''
    q = np.zeros_like(policy) # 存放动作价值
    c = np.zeros_like(policy) # 记录访问次数
    for _ in range(episode_num):
        # Play 1 round
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break # round end
        g = reward # reward
        for state, action in state_actions:
            # 增量更新
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q


def vis(data):
    '''状态价值可视化'''
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)


def monte_carlo_with_exploring_start(env, episode_num=10000):
    '''蒙特卡罗方法，带起始探索的回合更新'''

    # 初始策略
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 1] = 1.

    q = np.zeros_like(policy) # 保存动作价值
    c = np.zeros_like(policy) # 计数器

    for _ in range(episode_num):
        # 随机选择起始状态和起始动作
        state = (np.random.randint(12, 22),
                 np.random.randint(1, 11),
                 np.random.randint(2))
        action = np.random.randint(2)
        # 根据起始状态设置牌面
        env.reset()
        if state[2]: # 有A
            env.player = [1, state[0]-11] # A算11点
        else: # 没有A
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0]-10]
        env.dealer[0] = state[1]
        state_actions = []
        # 玩一回合
        while True:
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break # 回合结束
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
        g = reward # 回报

        for state, action in state_actions:
            # 评估
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            # 探索：取最大价值动作
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.

    return policy, q


def monte_carlo_with_soft(env, episode_num=10000, epsilon=0.1):
    '''柔性策略回合更新'''
    policy = np.ones((22, 11, 2, 2)) * 0.5 # 柔性策略
    q = np.zeros_like(policy) # 保存动作价值
    c = np.zeros_like(policy) # 计数器
    for _ in range(episode_num):
        # 玩一回合
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break # 回合结束
        g = reward # 回报
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            # 更新策略为柔性策略（以一定概率探索新策略）
            a = q[state].argmax()
            policy[state] = epsilon / 2.
            policy[state][a] += (1. - epsilon)
    return policy, q


def evaluate_monte_carlo_importance_sample(env, policy, behavior_policy,
                                           episode_num=10000):
    '''离线回合更新，重要性采样策略评估'''
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 用行为策略玩一回合
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n,
                                      p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break # round end
        g = reward # reward
        rho = 1. # 重要性采样比率
        # 注：
        # 普通平均值增量更新原理：
        # n * y(n) + x(n+1) = (n+1) * y(n+1) ——>
        # (n+1) * y(n) + x(n+1) = (n+1) * y(n+1) + y(n) ——>
        # y(n+1) - y(n) = (x(n+1) - y(n)) / (n+1)
        # 重要性采样加权平均值增量更新原理（权重w(i) = p(i) / sum(p(j))）
        # sum_n(p(i)) * y(n) + p(n+1) * x(n+1) = sum_n+1(p(i)) * y(n+1) ——>
        # sum_n(p(i)) * y(n) + y(n) * p(n+1) + p(n+1) * x(n+1) = sum_n+1(p(i)) * y(n+1) + y(n) * p(n+1) ——>
        # y(n+1) - y(n) = p(n+1) * (x(n+1) - y(n)) / sum_n+1(p(i))
        for state, action in reversed(state_actions): # 逆序？
            c[state][action] += rho
            q[state][action] += rho / c[state][action] * (g - q[state][action])
            rho *= (policy[state][action] / behavior_policy[state][action])
            if rho == 0:
                break # 提前终止
    return q


def monte_carlo_importance_resample(env, episode_num=10000):
    '''离线回合更新，柔性策略重要性采样回合更新求解最优策略'''

    policy = np.zeros((22, 11, 2, 2)) # 目标策略
    policy[:, :, :, 0] = 1.

    behavior_policy = np.ones_like(policy) * 0.5 # 柔性行为策略

    q = np.zeros_like(policy) # 动作价值
    c = np.zeros_like(policy) # 计数器
    for _ in range(episode_num):
        # 用行为策略玩一回合
        state_actions = []
        observation = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n,
                                      p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                break # 玩好了
        g = reward # 回报
        rho = 1. # 重要性采样比率
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            # 策略改进
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
            if a != action: # 提前终止
                break
            rho /= behavior_policy[state][action]
    return policy, q



if __name__ == '__main__':
    strt_tm = time.time()

    # BlackJack Game
    env = gym.make('Blackjack-v0')

    # Random policy
    def random_policy():
        return np.random.choice(2)
    N = 100
    rewards = [play_policy(env, random_policy) for _ in range(N)]
    print('Mean reward of {} rounds: {}'.format(N, np.mean(rewards)))

    # init one policy
    policy = np.zeros((32, 11, 2, 2))
    policy[20:, :, :, 0] = 1 # not continue when >= 20
    policy[:20, :, :, 1] = 1 # continue when < 20

    # MC方法策略评估
    q = evaluate_action_monte_carlo(env, policy) # 动作价值
    v = (q * policy).sum(axis=-1) # 动作价值加权期望即为状态价值
    vis(v)

    # MC带探索更新评估
    policy, q = monte_carlo_with_exploring_start(env)
    v = q.max(axis=-1)
    vis(policy.argmax(-1))
    vis(v)

    # MC柔性带探索更新评估
    policy, q = monte_carlo_with_soft(env)
    v = q.max(axis=-1)
    vis(policy.argmax(-1))
    vis(v)

    # 离线回合更新
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1 # >= 20 时收手
    policy[:20, :, :, 1] = 1 # < 20 时继续
    behavior_policy = np.ones_like(policy) * 0.5
    q = evaluate_monte_carlo_importance_sample(env, policy, behavior_policy)
    v = (q * policy).sum(axis=-1)
    vis(v)

    policy, q = monte_carlo_importance_resample(env)
    v = q.max(axis=-1)
    vis(policy.argmax(-1))
    vis(v)

    N = 100
    rewards = [play_policy2(env, policy) for _ in range(N)]
    print('Mean reward of {} rounds: {}'.format(N, np.mean(rewards)))


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
