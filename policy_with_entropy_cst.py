# Linear-Gaussian policy. Includes projection for entropy constraint. Entropy lower bound given by "entrop_profile.target_entropy".

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from policies import MLP


class MLPGaussianPolicy:
    def __init__(self, session, sizes, activations=None, init_sigma=1., full_cov=True, entrop_profile=None):
        self.sess = session

        with tf.variable_scope('policy'):
            # Building Gaussian policy
            # # mean
            self.mlp = MLP(sizes, activations)
            with tf.variable_scope('linear'):
                self.mean = tf.identity(self.mlp.out, name='mean')

                # # diag part of Cholesky
                act_dim = sizes[-1]
                self.logsigs_var = tf.Variable(np.log(init_sigma) * tf.ones([1, act_dim]), trainable=True, name='logstd') 
                self.logsigs = tf.tile(self.logsigs_var, [tf.shape(self.mlp.out)[0], 1])

                if act_dim > 1:
                    # # off-diag part of Cholesky
                    self.rest_chol = tf.Variable(tf.zeros([act_dim * (act_dim + 1) // 2 - act_dim]), trainable=full_cov,
                                                 name='restchol')
                    triu = list(zip(*np.tril_indices(act_dim)))
                    dind = list(zip(*np.diag_indices(act_dim)))
                    self.off_dind = [k for k in triu if k not in dind]
                    self.chol = tf.scatter_nd(self.off_dind, self.rest_chol, [act_dim, act_dim])

                    # # putting the two together
                    self.chol = tf.tile(self.chol[None], [tf.shape(self.logsigs)[0], 1, 1]) + tf.matrix_diag(tf.exp(self.logsigs))
                else:
                    self.chol = tf.matrix_diag(tf.exp(self.logsigs))


                # # rescaling cov if entropy constraint violated
                self.entrop_profile = entrop_profile
                if self.entrop_profile is not None:
                    ent = tf.reduce_sum(self.logsigs_var) + .5 * act_dim * tf.log(np.pi * 2 * np.e)
                    tent = self.entrop_profile.target_entropy
                    self.chol = tf.cond(ent < tent, lambda: self.chol * tf.exp((tent - ent) / act_dim), lambda: self.chol)
                self.covmat = tf.matmul(self.chol, self.chol, transpose_b=True)
                self.dist = tfp.distributions.MultivariateNormalTriL(self.mean, self.chol)

                # action tensor
                self.act_tensor = tf.identity(self.dist.sample(), name='act')

                # entropy
                self.entropy = tf.reduce_mean(self.dist.entropy())

                # action proba
                self.test_action = tf.placeholder(dtype=tf.float32, shape=[None, act_dim], name='test_action')
                self.log_prob = self.dist.log_prob(self.test_action)[:, None]
