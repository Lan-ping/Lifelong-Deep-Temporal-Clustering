from keras.engine.topology import Layer, InputSpec
import keras.backend as K

import numpy as np


class TCLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, dist_metric='eucl', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TCLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)
        self.clusters = None
        self.built = False

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        input_steps = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_steps, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_steps, input_dim), initializer='glorot_uniform', name='cluster_centers')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Student t-distribution kernel, probability of assigning encoded sequence i to cluster k.
            q_{ik} = (1 + dist(z_i, m_k)^2)^{-1} / normalization.

        Arguments:
            inputs: encoded input sequences, shape=(n_samples, timesteps, n_features)
        Return:
            q: soft labels for each sample. shape=(n_samples, n_clusters)
        """
        if self.dist_metric == 'eucl':
            distance = K.sum(K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)), axis=-1)
        elif self.dist_metric == 'cid':
            ce_x = K.sqrt(K.sum(K.square(inputs[:, 1:, :] - inputs[:, :-1, :]), axis=1))  # shape (n_samples, n_features)
            ce_w = K.sqrt(K.sum(K.square(self.clusters[:, 1:, :] - self.clusters[:, :-1, :]), axis=1))  # shape (n_clusters, n_features)
            ce = K.maximum(K.expand_dims(ce_x, axis=1), ce_w) / K.minimum(K.expand_dims(ce_x, axis=1), ce_w)  # shape (n_samples, n_clusters, n_features)
            ed = K.sqrt(K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2))  # shape (n_samples, n_clusters, n_features)
            distance = K.sum(ed * ce, axis=-1)  # shape (n_samples, n_clusters)
        elif self.dist_metric == 'cor':
            inputs_norm = (inputs - K.expand_dims(K.mean(inputs, axis=1), axis=1)) / K.expand_dims(K.std(inputs, axis=1), axis=1)  # shape (n_samples, timesteps, n_features)
            clusters_norm = (self.clusters - K.expand_dims(K.mean(self.clusters, axis=1), axis=1)) / K.expand_dims(K.std(self.clusters, axis=1), axis=1)  # shape (n_clusters, timesteps, n_features)
            pcc = K.mean(K.expand_dims(inputs_norm, axis=1) * clusters_norm, axis=2)  # Pearson correlation coefficients
            distance = K.sum(K.sqrt(2.0 * (1.0 - pcc)), axis=-1)  # correlation-based similarities, shape (n_samples, n_clusters)
        elif self.dist_metric == 'acf':
            raise NotImplementedError
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')
        q = 1.0 / (1.0 + K.square(distance) / self.alpha)
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'dist_metric': self.dist_metric}
        base_config = super(TCLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def add_k_centers(self, centers):
        # assert len(centers) == 3
        old_centers = self.get_weights()[0]
        assert old_centers.shape[1] == centers.shape[1] and old_centers.shape[2] == centers.shape[2]

        new_centers = np.concatenate([old_centers, centers])
        del self.trainable_weights[0]
        self.clusters = self.add_weight(shape=(new_centers.shape[0], new_centers.shape[1], new_centers.shape[2]), initializer='glorot_uniform',
                                        name='cluster_centers')
        self.set_weights([new_centers])
        self.n_clusters = self.n_clusters + centers.shape[0]
        return len(new_centers)

    def update_weights(self, new_centers):
        # assert len(new_centers) == 3
        old_centers = self.get_weights()[0]
        assert old_centers.shape[1] == new_centers.shape[1] and old_centers.shape[2] == new_centers.shape[2]

        if old_centers.shape[0] == new_centers.shape[0]:
            self.set_weights([new_centers])
        else:
            del self.trainable_weights[0]
            self.n_clusters = new_centers.shape[0]
            self.clusters = self.add_weight(shape=(new_centers.shape[0], new_centers.shape[1], new_centers.shape[2]),
                                            name='cluster_centers')
            self.set_weights([new_centers])
