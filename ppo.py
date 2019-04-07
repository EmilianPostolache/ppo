import numpy as np
from scipy.signal import lfilter
import tensorflow as tf


class PPO:
    """Class that implements the PPO learning algorithm.
    """
    def __init__(self, dim_obs, dim_act, gamma, lambda_, c, lr_policy, lr_valuef, logger):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.c = c
        self.policy = Policy(dim_obs, dim_act, c, lr_policy, logger)
        self.value_function = ValueFunction(dim_obs, lr_valuef, logger)

    def close(self):
        self.policy.close()
        self.value_function.close()

    def update(self, trajectories):
        advantages = self._compute_advantages(trajectories)
        discounted_returns = self._compute_discounted_returns(trajectories)
        observations = np.concatenate([trajectory['observations'] for trajectory in trajectories])
        actions = np.concatenate([trajectory['actions'] for trajectory in trajectories])
        self.policy.update(observations, actions, advantages)
        self.value_function.update(observations, discounted_returns)

    def _compute_advantages(self, trajectories):
        advantages = []
        for trajectory in trajectories:
            if self.gamma < 0.999:
                rewards = trajectory['rewards'] * (1 - self.gamma)
            else:
                rewards = trajectory['rewards']
            values = self.value_function.predict(trajectory['observations'])
            td_residuals = rewards - values + self.gamma * np.append(values[1:], 0)
            advantages.append(self._discount(td_residuals, self.gamma * self.lambda_))
        return np.concatenate(advantages)

    def _compute_discounted_returns(self, trajectories):
        discounted_returns = []
        for trajectory in trajectories:
            if self.gamma < 0.999:
                rewards = trajectory['rewards'] * (1 - self.gamma)
            else:
                rewards = trajectory['rewards']
            discounted_returns.append(self._discount(rewards, self.gamma))
        return np.concatenate(discounted_returns)

    @staticmethod
    def _discount(rewards, gamma):
        return lfilter([1.], [1., -gamma], rewards[::-1])[::-1]


class Policy:
    """ NN-based policy approximation
    """
    EPOCHS = 20
    KL_TARGET = 0.003

    """Policy network"""
    def __init__(self, dim_obs, dim_act, clip_range, lr, logger):
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.clip_range = clip_range
        self.lr = lr
        self.logger = logger
        self._build_graph()
        self._initialize()

    def _build_graph(self):
        # create a computational graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._placeholders()
            self._layers()
            self._logprob()
            self._kl()
            self._sample()
            self._loss()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        self.ph_obs = tf.placeholder(tf.float32, (None, self.dim_obs), 'obs')
        self.ph_act = tf.placeholder(tf.float32, (None, self.dim_act), 'act')
        self.ph_old_log_vars = tf.placeholder(tf.float32, (self.dim_act,), 'old_log_vars')
        self.ph_old_means = tf.placeholder(tf.float32, (None, self.dim_act,), 'old_means')
        self.ph_advantages = tf.placeholder(tf.float32, (None,), 'advantages')

    def _layers(self):
        # layer 1
        out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
                              name='layer1')
        # layer 2
        out = tf.layers.dense(out, 64, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                              name='layer2')
        # output layer (mean)
        self.means = tf.layers.dense(out, self.dim_act, name='means')
        self.log_vars = tf.get_variable('log_vars', (self.dim_act,), tf.float32)

    def _logprob(self):
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.ph_act - self.means) / tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.ph_old_log_vars)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.ph_act - self.ph_old_means) /
                                         tf.exp(self.ph_old_log_vars), axis=1)
        self.logp_old = logp_old

    def _kl(self):
        log_det_cov_old = tf.reduce_sum(self.ph_old_log_vars)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.ph_old_log_vars - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.ph_old_means) /
                                                     tf.exp(self.log_vars), axis=1) - self.dim_act)

    def _sample(self):
        self.op_sample = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal((self.dim_act,))

    def _loss(self):
        p_ratio = tf.exp(self.logp - self.logp_old)
        clipped_p_ratio = tf.clip_by_value(p_ratio, 1 - self.clip_range, 1 + self.clip_range)
        surrogate_loss = tf.minimum(self.ph_advantages * p_ratio, self.ph_advantages * clipped_p_ratio)
        self.loss = -tf.reduce_mean(surrogate_loss)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.op_train = optimizer.minimize(self.loss)

    def _initialize(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def close(self):
        self.sess.close()

    def update(self, obs, act, advantages):
        loss = None
        feed_dict = {self.ph_obs: obs,
                     self.ph_act: act,
                     self.ph_advantages: advantages}
        old_means, old_log_vars = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict.update({self.ph_old_means: old_means,
                          self.ph_old_log_vars: old_log_vars})
        for e in range(self.EPOCHS):
            self.sess.run(self.op_train, feed_dict)
            loss, kl = self.sess.run([self.loss, self.kl], feed_dict)
            # One could increment clipping range and reset it when exceed KL_TARGET
            if kl > self.KL_TARGET:
                break
        self.logger.log({'PolicyLoss': loss})

    def sample(self, obs):
        feed_dict = {self.ph_obs: obs}
        return self.sess.run(self.op_sample, feed_dict)


class ValueFunction:

    """Value Function network"""
    EPOCHS = 10

    def __init__(self, dim_obs, lr, logger):
        self.dim_obs = dim_obs
        self.lr = lr
        self.logger = logger
        self._build_graph()
        self._initialize()

    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._placeholders()
            self._layers()
            self._loss()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        self.ph_obs = tf.placeholder(tf.float32, (None, self.dim_obs), 'obs_valf')
        self.ph_val = tf.placeholder(tf.float32, (None,), 'val_valf')

    def _layers(self):
        # layer 1
        out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
                              name='layer1')
        # layer 2
        out = tf.layers.dense(out, 64, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                              name='layer2')
        # layer 3
        # output layer (mean)
        out = tf.layers.dense(out, 1, name='means')
        self.out = tf.squeeze(out)

    def _loss(self):
        self.loss = tf.reduce_mean(tf.square(self.out - self.ph_val))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.op_train = optimizer.minimize(self.loss)

    def _initialize(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def close(self):
        self.sess.close()

    def update(self, obs, returns):
        loss = None
        for e in range(self.EPOCHS):
            feed_dict = {self.ph_obs: obs,
                         self.ph_val: returns}
            self.sess.run(self.op_train, feed_dict)
            loss = self.sess.run(self.loss, feed_dict)
        self.logger.log({'ValFunctionLoss': loss})

    def predict(self, obs):
        feed_dict = {self.ph_obs: obs}
        pred = self.sess.run(self.out, feed_dict)
        return np.squeeze(pred)
