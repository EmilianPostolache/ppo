import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
from sklearn.utils import shuffle


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
            values = self.value_function.predict(trajectory['observations'])
            td_residuals = trajectory['rewards'] - values + self.gamma * np.append(values[1:], 0)
            advantages.append(self._discount(td_residuals, self.gamma * self.lambda_))
        return np.concatenate(advantages)

    def _compute_discounted_returns(self, trajectories):
        discounted_returns = []
        for trajectory in trajectories:
            discounted_returns.append(self._discount(trajectory['rewards'], self.gamma))
        return np.concatenate(discounted_returns)

    @staticmethod
    def _discount(rewards, gamma):
        return lfilter([1.], [1., -gamma], rewards[::-1])[::-1]


class Policy:

    LAYER_MULT = 10
    EPOCHS = 20

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
        out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
                              name='layer1')
        # layer 2
        out = tf.layers.dense(out, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                              name='layer2')
        # # layer 3
        # out = tf.layers.dense(out, int(self.dim_obs * self.LAYER_MULT / 2), tf.nn.tanh, name='layer3')
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
            loss = self.sess.run(self.loss, feed_dict)
        self.logger.log({'PolicyLoss': loss})

    def sample(self, obs):
        feed_dict = {self.ph_obs: obs}
        return self.sess.run(self.op_sample, feed_dict)


class ValueFunction:
    """ NN-based state-value function """
    HID1_MULT = 10

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = self.HID1_MULT
        self.epochs = 10
        self.lr = None  # learning rate set in _build_graph()
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
            hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
            hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
            hid2_size = int(np.sqrt(hid1_size * hid3_size))
            # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
                  .format(hid1_size, hid2_size, hid3_size, self.lr))
            # 3 hidden layers with tanh activations
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat) / np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))  # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    # """Value Function network"""
    # LAYER_MULT = 5
    # EPOCHS = 10
    #
    # def __init__(self, dim_obs, lr, logger):
    #     self.dim_obs = dim_obs
    #     self.lr = lr
    #     self.logger = logger
    #     self._build_graph()
    #     self._initialize()
    #
    # def _build_graph(self):
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         self._placeholders()
    #         self._layers()
    #         self._loss()
    #         self.init = tf.global_variables_initializer()
    #
    # def _placeholders(self):
    #     self.ph_obs = tf.placeholder(tf.float32, (None, self.dim_obs), 'obs_valf')
    #     self.ph_val = tf.placeholder(tf.float32, (None,), 'val_valf')
    #
    # def _layers(self):
    #     # layer 1
    #     out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
    #                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
    #                           name='layer1')
    #     # layer 2
    #     out = tf.layers.dense(out, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
    #                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
    #                           name='layer2')
    #     # layer 3
    #     #  out = tf.layers.dense(out, int(self.dim_obs * self.LAYER_MULT / 2), tf.nn.tanh, name='layer3')
    #     # output layer (mean)
    #     out = tf.layers.dense(out, 1, name='means')
    #     self.out = tf.squeeze(out)
    #
    # def _loss(self):
    #     self.loss = tf.reduce_mean(tf.square(self.out - self.ph_val))
    #     optimizer = tf.train.AdamOptimizer(self.lr)
    #     self.op_train = optimizer.minimize(self.loss)
    #
    # def _initialize(self):
    #     self.sess = tf.Session(graph=self.graph)
    #     self.sess.run(self.init)
    #
    # def close(self):
    #     self.sess.close()
    #
    # def update(self, obs, returns):
    #     loss = None
    #     for e in range(self.EPOCHS):
    #         feed_dict = {self.ph_obs: obs,
    #                      self.ph_val: returns}
    #         self.sess.run(self.op_train, feed_dict)
    #         loss = self.sess.run(self.loss, feed_dict)
    #     self.logger.log({'ValFunctionLoss': loss})
    #
    # def predict(self, obs):
    #     feed_dict = {self.ph_obs: obs}
    #     pred = self.sess.run(self.out, feed_dict)
    #     return np.squeeze(pred)
