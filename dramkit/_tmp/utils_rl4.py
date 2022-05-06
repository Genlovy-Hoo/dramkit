# -*- coding: utf-8 -*-

import time
import gym
import numpy as np


def play_policy(env, policy, render=False):
    '''冰面滑行单回合'''
    total_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = np.random.choice(env.action_space.n, p=policy[observation])
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def v2q(env, v, s=None, gamma=1):
    '''
    策略评估，根据状态价值函数计算动作价值函数
    '''
    if s is not None: # 针对单个状态求解
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1-done))
    else: # 针对所有状态求解
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


def evaluate_policy(env, policy, gamma=1, tol=1e-6):
    '''策略评估，计算策略状态价值函数'''
    v = np.zeros(env.unwrapped.nS)
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vs = sum(policy[s] * v2q(env, v, s, gamma)) # 更新状态价值函数
            delta = max(delta, abs(v[s]-vs)) # 误差更新
            v[s] = vs # 更新状态价值函数
        if delta < tol:
            break
    return v


def improve_policy(env, v, policy, gamma=1):
    '''策略改进'''
    optimal = True
    for s in range(env.unwrapped.nS):
        q = v2q(env, v, s, gamma)
        a = np.argmax(q)
        if policy[s][a] != 1.0:
            optimal = False
            policy[s] = 0
            policy[s][a] = 1
    return optimal


def iterate_policy(env, gamma=1, tol=1e-6):
    '''策略迭代'''
    # 策略初始化
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tol) # 策略评估
        if improve_policy(env, v, policy): # 策略改进
            break
    return policy, v


def iterate_value(env, gamma=1, tol=1e-6):
    '''价值迭代'''
    v = np.zeros(env.unwrapped.nS) # 价值函数初始化
    while True:
        delta = 0
        for s in range(env.unwrapped.nS):
            vtmp = v2q(env, v, s, gamma)
            vmax = max(vtmp) # 更新价值函数
            delta = max(delta, abs(v[s]-vmax))
            v[s] = vmax
        if delta < tol:
            break
        # 计算最优策略
        policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            a = np.argmax(v2q(env, v, s, gamma))
            policy[s][a] = 1.0
    return policy, v


if __name__ == '__main__':
    strt_tm = time.time()

    # 冰面滑行
    env = gym.make('FrozenLake-v0')
    env = env.unwrapped

    # 随机策略
    random_policy = \
        np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
    N = 100
    rewards = [play_policy(env, random_policy) for _ in range(0, N)]
    print('随机策略{}次平均奖励：{}'.format(N, np.mean(rewards)))

    v_random = evaluate_policy(env, random_policy)
    print('随机策略状态价值函数：\n{}'.format(v_random.reshape(4, 4)))
    q_random = v2q(env, v_random)
    print('随机策略动作价值函数：\n{}'.format(q_random))

    policy = random_policy.copy()
    optimal = improve_policy(env, v_random, policy)
    if optimal:
        print('无更新，最优策略为：\n{}'.format(policy))
    else:
        print('有更新，最优策略为：\n{}'.format(policy))

    # 策略迭代求最优策略
    policy_pi, v_pi = iterate_policy(env)
    print('状态价值函数 = \n{}'.format(v_pi.reshape(4, 4)))
    print('最优策略 = \n{}'.format(np.argmax(policy_pi, axis=1).reshape(4, 4)))

    # 价值迭代求最优策略
    policy_vi, v_vi = iterate_value(env)
    print('状态价值函数 = \n{}'.format(v_vi.reshape(4, 4)))
    print('最优策略 = \n{}'.format(np.argmax(policy_vi, axis=1).reshape(4, 4)))

    print('策略迭代最优策略{}次平均奖励：'.format(N) + \
          '{}'.format(np.mean([play_policy(env, policy_pi) for _ in range(N)])))
    print('价值迭代最优策略{}次平均奖励：'.format(N) + \
      '{}'.format(np.mean([play_policy(env, policy_vi) for _ in range(N)])))


    print('used time: {}s.'.format(round(time.time()-strt_tm, 6)))
