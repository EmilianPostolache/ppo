#! /usr/bin/env python3
""" This module contains the main loop of the training
simulation. It is separated from the algorithm that
implements the learning process, so it can be used with
a wide range of RL algorithms.

Written by Emilian Postolache (github.com/EmilianPostolache)

Based on an implementation by Patrick Coady (pat-coady.github.io)
"""
import time

import gym
import numpy as np
from gym import wrappers
import datetime
import os
from util import Logger, GracefulExit, VecScaler
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


# Tricks: - scale observations   !
#         - add a time step feature   !

def run_episode(env, policy, scaler):
    observations, actions, rewards, unscaled_obs = [], [], [], []
    observation = env.reset()
    scale, offset = scaler.get()
    step = 0.0
    scale[-1] = 1.0
    offset[-1] = 0.0
    done = False
    while not done:
        observation = np.append(observation, step)
        observation = observation.reshape(1, -1)
        unscaled_obs.append(observation)
        observation = (observation - offset) * scale
        observations.append(observation)
        action = policy.sample(observation)
        observation, reward, done, _ = env.step(action)
        actions.append(action.reshape(1, -1))
        rewards.append(reward)
    return (np.concatenate(observations), np.concatenate(actions), np.array(rewards),
            np.concatenate(unscaled_obs))


def run_batch(env, policy, batch_size, scaler):
    trajectories = []
    steps = 0
    for episode in range(batch_size):
        observations, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        trajectory = {'observations': observations,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        steps += observations.shape[0]
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)
    mean_return = np.mean([trajectory['rewards'].sum() for trajectory in trajectories])
    return trajectories, steps, mean_return


def train():
    g_exit = GracefulExit()
    timestamp = datetime.datetime.utcnow().strftime(TIMESTAMP_FORMAT)
    logger = Logger(ENV_NAME, timestamp)
    env = gym.make(ENV_NAME)
    dim_obs = env.observation_space.shape[0] + 1
    dim_act = env.action_space.shape[0]
    scaler = VecScaler(dim_obs)
    rec_dir = os.path.join(REC_DIR, ENV_NAME, timestamp)
    env = gym.wrappers.Monitor(env, rec_dir, force=True)
    agent = PPO(dim_obs, dim_act, GAMMA, LAMBDA, CLIP_RANGE, LR_POLICY,
                LR_VALUE_F, logger)
    run_batch(env, agent.policy, 5, scaler)
    episode = 0
    while episode < NUM_EPISODES:
        batch_size = min(MAX_BATCH, NUM_EPISODES - episode)
        trajectories, steps, mean_return = run_batch(env, agent.policy, batch_size, scaler)
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
