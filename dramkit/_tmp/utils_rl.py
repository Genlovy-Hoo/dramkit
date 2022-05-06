# -*- coding: utf-8 -*-

import time
import gym
from gym import envs
import numpy as np

#%%
if __name__ == '__main__':
    strt_tm = time.time()

    #%%
    # 查看当前gym已经注册的环境
    env_specs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in env_specs]
    print('当前gym已经注册的环境:')
    print(env_ids)

    #%%
    # 查看任务环境
    env_id = 'CartPole-v0'
    # env_id = 'MountainCar-v0'
    env = gym.make(env_id)
    O = env.observation_space # 观测空间
    A = env.action_space # 动作空间
    print('\n')
    print('观测空间 of {}:\n{}'.format(env_id, O))
    print('动作空间 of {}:\n{}'.format(env_id, A))

    observation = env.reset()
    print('init observation: {}'.format(observation))

    # 任务示例
    for _ in range(0, 100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
        time.sleep(0.002)
    env.close()

    #%%
    # MountainCar任务示例
    env_id = 'MountainCar-v0'
    env = gym.make(env_id)
    print('\n')
    O = env.observation_space
    print('观测空间 of {}: \n{}'.format(env_id, O))
    print('动作空间 of {}: \n{}'.format(env_id, env.action_space))
    print('观测范围 of {}: \n{}~{}'.format(env_id, O.low, O.high))

    # 自定义智能体测试
    class BespokeAgent:
        def __init__(self, env):
            pass

        def decide(self, observation):
            '''决策'''
            position, velocity = observation
            lb = min(-0.09 * (position+0.25)**2 + 0.03,
                     0.3 * (position+0.9)**4 - 0.008)
            ub = -0.07 * (position*0.38)**2 + 0.06
            if lb < velocity < ub:
                action = 2
            else:
                action = 0
            return action

        def learn(self, *args):
            '''学习'''
            pass

    agent = BespokeAgent(env)

    # 交互一个回合测试
    def play_montcarlo(env, agent, render=True, train=False):
        episode_reward = 0 # 记录一个回合总奖励

        observation = env.reset() # 重置任务,开始新回合
        while True:

            if render:
                env.render()

            action = agent.decide(observation)
            observation_, reward, done, _ = env.step(action) # 执行动作
            episode_reward += reward # 奖励更新

            # 学习
            if train:
                agent.learn(observation, action, reward, done)

            if done:
                break

            observation = observation_ # 环境观测更新

        return episode_reward

    env.seed(5262)
    episode_reward = play_montcarlo(env, agent, render=True)
    print('\n单回合奖励: {}'.format(episode_reward))
    time.sleep(2)
    env.close()

    # 测试100个回合
    render = False
    rewards = [play_montcarlo(env, agent, render=render) for _ in range(100)]
    print('平均回合奖励:{}'.format(round(np.mean(rewards), 4)))
    if render:
        env.close()

    #%%
    print('\nused time: {}s.'.format(round(time.time()-strt_tm, 6)))
