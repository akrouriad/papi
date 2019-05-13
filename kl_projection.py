# Projection for linear-Gaussian policies. Corresponding to Alg. 2 in paper "Projections for Approximate Policy Iteration Algorithms".

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class UpdateProjGaussMean:
    def __init__(self, session, policy, a_dim, e_kl=.05):
        self.sess, self.pol, self.e_kl = session, policy, e_kl
        
        # data from old policy
        self.old_mean = tf.placeholder(dtype=tf.float32, shape=[None, a_dim], name='old_mean')
        self.old_chol = tf.placeholder(dtype=tf.float32, shape=[None, a_dim, a_dim], name='old_chol')
        self.intermediate_mean = tf.placeholder(dtype=tf.float32, shape=[None, a_dim], name='intermediate_mean')

        self.cq = tf.matmul(self.old_chol, self.old_chol, transpose_b=True)
        rep_eye = tf.tile(tf.eye(a_dim)[None], [tf.shape(self.old_chol)[0], 1, 1])
        invt_cq = tf.matrix_triangular_solve(self.old_chol, rep_eye)
        self.pq = tf.matmul(invt_cq, invt_cq, transpose_a=True)
        self.logdetpq = -2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.old_chol)), axis=-1, keepdims=True)
        self.old_dist = tfp.distributions.MultivariateNormalTriL(self.old_mean, self.old_chol)

        # KL violation
        self.m = self.dm = UpdateProjGaussMean.mean_diff(self.pol.mean, self.old_mean, self.pq)
        self.r = self.dr = UpdateProjGaussMean.rot_diff(self.pol.covmat, self.pq)
        self.e = self.de = UpdateProjGaussMean.entropy_diff(self.pol.chol, self.logdetpq)
        self.inter_mean_diff = UpdateProjGaussMean.mean_diff(self.intermediate_mean, self.old_mean, self.pq)
        self.mm = tf.minimum(self.m, self.inter_mean_diff)
        self.init_kl = self.r + self.m + self.e
        self.tf_og_kl = self.pol.dist.kl_divergence(self.old_dist)

        # projecting: matrix rotation/rescaling
        self.eta_rot = (e_kl - self.mm) / tf.maximum(self.r + self.m + self.e, 1e-16)
        ncov = (1 - self.eta_rot) * self.cq + self.eta_rot * self.pol.covmat
        self.do_intercov = self.m + self.r + self.e > e_kl + 1e-6
        self.chol, cov = tf.cond(self.do_intercov, true_fn=lambda: (tf.cholesky(ncov), ncov), false_fn=lambda: (self.pol.chol, self.pol.covmat))
        self.r, self.e = tf.cond(self.do_intercov,
                                 true_fn=lambda: (UpdateProjGaussMean.rot_diff(cov, self.pq), UpdateProjGaussMean.entropy_diff(self.chol, self.logdetpq)),
                                 false_fn=lambda: (self.r, self.e))

        # projecting: mean interpolation
        self.tmean = tf.maximum(e_kl - self.e - self.r, 0)
        self.a = UpdateProjGaussMean.mean_diff(self.pol.mean, self.intermediate_mean, self.pq)
        self.b = UpdateProjGaussMean.cross_mul(self.pol.mean - self.intermediate_mean, self.pq, self.intermediate_mean - self.old_mean)
        self.c = self.inter_mean_diff - self.tmean
        self.eta_mean = (-self.b + tf.sqrt(tf.maximum(self.b * self.b - self.a * self.c, 1e-16))) / tf.maximum(self.a, 1e-16)
        nmean = (1 - self.eta_mean) * self.intermediate_mean + self.eta_mean * self.pol.mean
        self.do_shift = self.m + self.r + self.e > e_kl + 1e-6
        mean = tf.cond(self.do_shift, true_fn=lambda: nmean, false_fn=lambda: self.pol.mean)
        self.m = tf.cond(self.do_shift, true_fn=lambda: UpdateProjGaussMean.mean_diff(mean, self.old_mean, self.pq), false_fn=lambda: self.m)
        self.final_kl = self.r + self.m + self.e

        # building distributions
        self.proj_dist = tfp.distributions.MultivariateNormalTriL(mean, self.chol)
        self.tf_kl = self.proj_dist.kl_divergence(self.old_dist)
        self.log_prob = tf.transpose(self.proj_dist.log_prob(self.pol.test_action_m))[:, :, None]

    @staticmethod
    def mean_diff(mu, oldmu, pq):
        mud = mu - oldmu
        return UpdateProjGaussMean.cross_mul(mud, pq, mud)

    @staticmethod
    def cross_mul(l, mat, r):
        return .5 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(l[:, :, None] * mat, 1) * r, 1, keepdims=True))

    @staticmethod
    def rot_diff(cov, pq):
        dim = tf.cast(tf.shape(cov)[-1], dtype=tf.float32)
        return .5 * tf.reduce_mean((tf.trace(tf.matmul(pq, cov)) - dim))

    @staticmethod
    def entropy_diff(chol, logdetp):
        return .5 * tf.reduce_mean((-logdetp - 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), axis=1, keepdims=True)))
