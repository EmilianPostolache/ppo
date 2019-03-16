import tensorflow as tf
import numpy as np

class PolicyNetwork:

    def __init__(self, dim_s, dim_a,
                 dim_hidden=64, dim_batch=32, lr=1e-3,
                 gamma=0.99):
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_hidden = dim_hidden
        self.lr = lr
        self.dim_batch = dim_batch
        self.gamma = gamma
        self._build_model()

    def _loss(self, mean, precision, G):
        ...

    def _build_model(self):
        tf.reset_default_graph()

        input = tf.placeholder(tf.float32, shape=(None, self.dim_s))
        G = tf.placeholder(tf.float32, shape=(None, 1))

        W0 = tf.get_variable('W0', shape=(self.dim_s, self.dim_hidden), dtype=tf.float32)
        b0 = tf.get_variable('b0', shape=(self.dim_hidden,), dtype=tf.float32)

        W1 = tf.get_variable('W1', shape=(self.dim_hidden, self.dim_hidden), dtype=tf.float32)
        b1 = tf.get_variable('b1', shape=(self.dim_hidden,), dtype=tf.float32)

        W2 = tf.get_variable('W2', shape=(self.dim_hidden, self.dim_a), dtype=tf.float32)
        b2 = tf.get_variable('b2', shape=(self.dim_a), dtype=tf.float32)

        hidden0 = tf.nn.tanh(input @ W0 + b0)
        hidden1 = tf.nn.tanh(hidden0 @ W1 + b1)
        mean = hidden1 @ W2 + b2
        precision = tf.get_variable('prec', shape=(self.dim_a,), dtype=tf.float32)

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer.minimize(self._loss(mean, precision, G))

    def update(self):
        ...

if __name__ == '__main__':
