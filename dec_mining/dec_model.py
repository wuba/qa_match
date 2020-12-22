# -*- coding: utf-8 -*-
"""
modified DEC model
"""
import collections
import re
import sys
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
sys.path.append("..")
from sptm import models
import data_utils


""" modified DEC model
    References
    ----------
    https://github.com/HaebinShin/dec-tensorflow/blob/master/dec/model.py
"""
class DEC:
    def __init__(self, params, other_params):
        # pass n_clusters or external_cluster_center
        if params.external_cluster_center == "":
            self.n_cluster = params.n_clusters
            self.kmeans = KMeans(n_clusters=params.n_clusters, n_init=10)
        else:
            # get cluster_center_embedding, n_cluster  from external file
            self.external_cluster_center_vec, self.n_cluster = \
                data_utils.load_spec_centers(params.external_cluster_center)

        self.embedding_dim = params.embedding_dim
        # load SPTM pretrained model
        model = models.create_finetune_classification_model(params, other_params)
        self.pretrained_model = model
        self.alpha = params.alpha

        # mu: cluster center
        self.mu = tf.Variable(tf.zeros(shape=(self.n_cluster, self.embedding_dim)),
                              name="mu")  # [n_class, emb_dim]

        self.z = model.max_pool_output  # [batch, emb_dim]
        with tf.name_scope("distribution"):
            self.q = self._soft_assignment(self.z, self.mu)  # [, n_class]
            self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))  # [, n_class]
            self.pred = tf.argmax(self.q, axis=1)
            self.pred_prob = tf.reduce_max(self.q, axis=1)

        with tf.name_scope("dec-train"):
            self.loss = self._kl_divergence(self.p, self.q)
            self.global_step_op = tf.train.get_or_create_global_step()
            self.lr = params.learning_rate
            warmup_steps = params.warmup_steps
            warmup_lr = (self.lr * tf.cast(self.global_step_op, tf.float32)
                         / tf.cast(warmup_steps, tf.float32))
            self.warmup_learning_rate_op = \
                tf.cond(self.global_step_op < warmup_steps, lambda: warmup_lr, lambda: self.lr)
            self.optimizer = tf.train.AdamOptimizer(self.warmup_learning_rate_op)
            self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step_op)

    def get_assign_cluster_centers_op(self, features):
        # init mu
        tf.logging.info("Kmeans train start.")
        kmeans = self.kmeans.fit(features)
        tf.logging.info("Kmeans train end.")
        return tf.assign(self.mu, kmeans.cluster_centers_)

    # emb [batch, emb_dim]   centroid [n_class, emb_dim]
    def _soft_assignment(self, embeddings, cluster_centers):
        """Implemented a soft assignment as the  probability of assigning sample i to cluster j.

        Args:
            embeddings: (num_points, dim)
            cluster_centers: (num_cluster, dim)

        Return:
            q_i_j: (num_points, num_cluster)
        """
        def _pairwise_euclidean_distance(a, b):
            # p1 [batch, n_class]
            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
                tf.ones(shape=(1, self.n_cluster))
            )
            # p2 [batch, n_class]
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
                tf.ones(shape=(tf.shape(a)[0], 1)),
                transpose_b=True
            ))
            # [batch, n_class]
            res = tf.sqrt(
                tf.abs(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True)))

            return res

        dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
        q = 1.0 / (1.0 + dist ** 2 / self.alpha) ** ((self.alpha + 1.0) / 2.0)
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def cluster_acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
