#! /usr/bin/env python3
""" This module contains the main loop of the training
simulation. It is separated from the algorithm that
implements the learning process, so it can be used with
a wide range of RL algorithms.

Written by Emilian Postolache (github.com/EmilianPostolache)

Based on an implementation by Patrick Coady (pat-coady.github.io)
"""

import gym
import numpy as np
from gym import wrappers
import datetime
import os
from util import Logger, GracefulExit
from ppo import PPO

ENV_NAME = 'Humanoid-v3'
REC_DIR = 'recordings'
LOG_DIR = 'logs'
TIMESTAMP_FORMAT = '%b-%d_%H:%M:%S'

# Hyperparameters
GAMMA = 0.995
LAMBDA = 0.98
NUM_EPISODES = 100000
MAX_BATCH = 20
CLIP_RANGE = 0.2
LR_POLICY = 3e-4
LR_VALUE_F = 3e-4


def run_episode(env, policy):
    observations, actions, rewards = [], [], []
    observation = env.reset()
    done = False
    while not done:
        observation = observation.reshape(1, -1)
        observations.append(observation)
        action = policy.sample(observation)
        observation, reward, done, _ = env.step(action)
        actions.append(action.reshape(1, -1))
        rewards.append(reward)
    return np.concatenate(observations), np.concatenate(actions), np.array(rewards)


def run_batch(env, policy, batch_size):
    trajectories = []
    steps = 0
    for episode in range(batch_size):
        observations, actions, rewards = run_episode(env, policy)
        trajectory = {'observations': observations,
                      'actions': actions,
                      'rewards': rewards}
        steps += observations.shape[0]
        trajectories.append(trajectory)
    mean_return = np.mean([trajectory['rewards'].sum() for trajectory in trajectories])
    return trajectories, steps, mean_return


def train():
    g_exit = GracefulExit()
    timestamp = datetime.datetime.utcnow().strftime(TIMESTAMP_FORMAT)
    logger = Logger(ENV_NAME, timestamp)
    env = gym.make(ENV_NAME)
    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]
    rec_dir = os.path.join(REC_DIR, ENV_NAME, timestamp)
    env = gym.wrappers.Monitor(env, rec_dir, force=True)
    agent = PPO(dim_obs, dim_act, GAMMA, LAMBDA, CLIP_RANGE, LR_POLICY,
                LR_VALUE_F, logger)
    episode = 0
    while episode < NUM_EPISODES:
        batch_size = min(MAX_BATCH, NUM_EPISODES - episode)
        trajectories, steps, mean_return = run_batch(env, agent.policy, batch_size)
        episode += batch_size
        logger.log({'_time': datetime.datetime.utcnow().strftime(TIMESTAMP_FORMAT),
                    '_episode': episode,
                    'steps': steps,
                    '_mean_return': mean_return})
        agent.update(trajectories)
        logger.write()
        if g_exit.exit:
            break
    agent.close()
    logger.close()


if __name__ == '__main__':
    train()
