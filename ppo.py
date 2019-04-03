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
    """ NN-based policy approximation """

    def __init__(self,  dim_obs, dim_act, clip_range, lr, logger):
        # self, obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """
        self.logger = logger
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = 0.003
        self.hid1_mult = 10
        self.policy_logvar = 1.0
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = dim_obs
        self.act_dim = dim_act
        self.clipping_range = [clip_range, clip_range]
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """

        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = self.act_dim * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_size)  # 9e-4 empirically determined
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
        self.means = tf.layers.dense(out, self.act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / hid3_size)), name="means")
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_size, hid2_size, hid3_size, self.lr, logvar_speed))

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self):
        """
        Add to Graph:
            1. KL divergence between old and new distributions
            2. Entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """ Sample from distribution, given observation """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        if self.clipping_range is not None:
            print('setting up loss with clipping objective')
            pg_ratio = tf.exp(self.logp - self.logp_old)
            clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0], 1 + self.clipping_range[1])
            surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                        self.advantages_ph * clipped_pg_ratio)
            self.loss = -tf.reduce_mean(surrogate_loss)
        else:
            print('setting up loss with KL penalty')
            loss1 = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.loss = loss1 + loss2 + loss3
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

    def close(self):
        """ Close TensorFlow session """
        self.sess.close()

    # LAYER_MULT = 10
    # EPOCHS = 20
    #
    # """Policy network"""
    # def __init__(self, dim_obs, dim_act, clip_range, lr, logger):
    #     self.dim_obs = dim_obs
    #     self.dim_act = dim_act
    #     self.clip_range = clip_range
    #     self.lr = lr
    #     self.logger = logger
    #     self._build_graph()
    #     self._initialize()
    #
    # def _build_graph(self):
    #     # create a computational graph
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         self._placeholders()
    #         self._layers()
    #         self._logprob()
    #         self._sample()
    #         self._loss()
    #         self.init = tf.global_variables_initializer()
    #
    # def _placeholders(self):
    #     self.ph_obs = tf.placeholder(tf.float32, (None, self.dim_obs), 'obs')
    #     self.ph_act = tf.placeholder(tf.float32, (None, self.dim_act), 'act')
    #     self.ph_old_log_vars = tf.placeholder(tf.float32, (self.dim_act,), 'old_log_vars')
    #     self.ph_old_means = tf.placeholder(tf.float32, (None, self.dim_act,), 'old_means')
    #     self.ph_advantages = tf.placeholder(tf.float32, (None,), 'advantages')
    #
    # def _layers(self):
    #     # layer 1
    #     out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
    #                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
    #                           name='layer1')
    #     # layer 2
    #     out = tf.layers.dense(out, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT, tf.nn.tanh,
    #                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
    #                           name='layer2')
    #     # # layer 3
    #     # out = tf.layers.dense(out, int(self.dim_obs * self.LAYER_MULT / 2), tf.nn.tanh, name='layer3')
    #     # output layer (mean)
    #     self.means = tf.layers.dense(out, self.dim_act, name='means')
    #     self.log_vars = tf.get_variable('log_vars', (self.dim_act,), tf.float32)
    #
    # def _logprob(self):
    #     logp = -0.5 * tf.reduce_sum(self.log_vars)
    #     logp += -0.5 * tf.reduce_sum(tf.square(self.ph_act - self.means) / tf.exp(self.log_vars), axis=1)
    #     self.logp = logp
    #
    #     logp_old = -0.5 * tf.reduce_sum(self.ph_old_log_vars)
    #     logp_old += -0.5 * tf.reduce_sum(tf.square(self.ph_act - self.ph_old_means) /
    #                                      tf.exp(self.ph_old_log_vars), axis=1)
    #     self.logp_old = logp_old
    #
    # def _sample(self):
    #     self.op_sample = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal((self.dim_act,))
    #
    # def _loss(self):
    #     p_ratio = tf.exp(self.logp - self.logp_old)
    #     clipped_p_ratio = tf.clip_by_value(p_ratio, 1 - self.clip_range, 1 + self.clip_range)
    #     surrogate_loss = tf.minimum(self.ph_advantages * p_ratio, self.ph_advantages * clipped_p_ratio)
    #     self.loss = -tf.reduce_mean(surrogate_loss)
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
    # def update(self, obs, act, advantages):
    #     loss = None
    #     feed_dict = {self.ph_obs: obs,
    #                  self.ph_act: act,
    #                  self.ph_advantages: advantages}
    #     old_means, old_log_vars = self.sess.run([self.means, self.log_vars], feed_dict)
    #     feed_dict.update({self.ph_old_means: old_means,
    #                       self.ph_old_log_vars: old_log_vars})
    #     for e in range(self.EPOCHS):
    #         self.sess.run(self.op_train, feed_dict)
    #         loss = self.sess.run(self.loss, feed_dict)
    #     self.logger.log({'PolicyLoss': loss})
    #
    # def sample(self, obs):
    #     feed_dict = {self.ph_obs: obs}
    #     return self.sess.run(self.op_sample, feed_dict)


class ValueFunction:
    # """ NN-based state-value function """
    # HID1_MULT = 10
    #
    # def __init__(self, obs_dim, lr, logger):
    #     """
    #     Args:
    #         obs_dim: number of dimensions in observation vector (int)
    #         hid1_mult: size of first hidden layer, multiplier of obs_dim
    #     """
    #     self.logger = logger
    #     self.replay_buffer_x = None
    #     self.replay_buffer_y = None
    #     self.obs_dim = obs_dim
    #     self.hid1_mult = self.HID1_MULT
    #     self.epochs = 10
    #     self.lr = None  # learning rate set in _build_graph()
    #     self._build_graph()
    #     self.sess = tf.Session(graph=self.g)
    #     self.sess.run(self.init)
    #
    # def _build_graph(self):
    #     """ Construct TensorFlow graph, including loss function, init op and train op """
    #     self.g = tf.Graph()
    #     with self.g.as_default():
    #         self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
    #         self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
    #         # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
    #         hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
    #         hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
    #         hid2_size = int(np.sqrt(hid1_size * hid3_size))
    #         # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
    #         self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
    #         print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
    #               .format(hid1_size, hid2_size, hid3_size, self.lr))
    #         # 3 hidden layers with tanh activations
    #         out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
    #                               kernel_initializer=tf.random_normal_initializer(
    #                                   stddev=np.sqrt(1 / self.obs_dim)), name="h1")
    #         out = tf.layers.dense(out, hid2_size, tf.tanh,
    #                               kernel_initializer=tf.random_normal_initializer(
    #                                   stddev=np.sqrt(1 / hid1_size)), name="h2")
    #         out = tf.layers.dense(out, hid3_size, tf.tanh,
    #                               kernel_initializer=tf.random_normal_initializer(
    #                                   stddev=np.sqrt(1 / hid2_size)), name="h3")
    #         out = tf.layers.dense(out, 1,
    #                               kernel_initializer=tf.random_normal_initializer(
    #                                   stddev=np.sqrt(1 / hid3_size)), name='output')
    #         self.out = tf.squeeze(out)
    #         self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
    #         optimizer = tf.train.AdamOptimizer(self.lr)
    #         self.train_op = optimizer.minimize(self.loss)
    #         self.init = tf.global_variables_initializer()
    #     self.sess = tf.Session(graph=self.g)
    #     self.sess.run(self.init)
    #
    # def update(self, x, y):
    #     """ Fit model to current data batch + previous data batch
    #
    #     Args:
    #         x: features
    #         y: target
    #         logger: logger to save training loss and % explained variance
    #     """
    #     num_batches = max(x.shape[0] // 256, 1)
    #     batch_size = x.shape[0] // num_batches
    #     y_hat = self.predict(x)  # check explained variance prior to update
    #     old_exp_var = 1 - np.var(y - y_hat) / np.var(y)
    #     if self.replay_buffer_x is None:
    #         x_train, y_train = x, y
    #     else:
    #         x_train = np.concatenate([x, self.replay_buffer_x])
    #         y_train = np.concatenate([y, self.replay_buffer_y])
    #     self.replay_buffer_x = x
    #     self.replay_buffer_y = y
    #     for e in range(self.epochs):
    #         x_train, y_train = shuffle(x_train, y_train)
    #         for j in range(num_batches):
    #             start = j * batch_size
    #             end = (j + 1) * batch_size
    #             feed_dict = {self.obs_ph: x_train[start:end, :],
    #                          self.val_ph: y_train[start:end]}
    #             _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
    #     y_hat = self.predict(x)
    #     loss = np.mean(np.square(y_hat - y))  # explained variance after update
    #     exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func
    #
    #     self.logger.log({'ValFuncLoss': loss,
    #                 'ExplainedVarNew': exp_var,
    #                 'ExplainedVarOld': old_exp_var})
    #
    # def predict(self, x):
    #     """ Predict method """
    #     feed_dict = {self.obs_ph: x}
    #     y_hat = self.sess.run(self.out, feed_dict=feed_dict)
    #
    #     return np.squeeze(y_hat)
    #
    # def close(self):
    #     """ Close TensorFlow session """
    #     self.sess.close()

    """Value Function network"""
    LAYER_MULT = 5
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
        out = tf.layers.dense(self.ph_obs, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.dim_obs)),
                              name='layer1')
        # layer 2
        out = tf.layers.dense(out, 64, tf.nn.tanh,  # self.dim_obs * self.LAYER_MULT,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                              name='layer2')
        # layer 3
        #  out = tf.layers.dense(out, int(self.dim_obs * self.LAYER_MULT / 2), tf.nn.tanh, name='layer3')
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
