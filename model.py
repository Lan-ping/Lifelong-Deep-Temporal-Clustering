"""
Implementation of the Deep Temporal Clustering model
Main file

@author Florent Forest (FlorentF9)
"""

# Utilities
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import csv
from time import time

# Keras
from keras.models import Model
from keras.layers import Input
from keras.models import clone_model
import keras.backend as K

# scikit-learn
from sklearn.cluster import AgglomerativeClustering, KMeans

# Dataset helper function
from data_util.metrics import *
from data_util import tsdistances
from data_util.datasets import data_cluster_divide

# LDTC components
from layer_util.TCLayer import TCLayer
from layer_util.CTAE import temporal_autoencoder_v2

import random


class LDTC:

    def __init__(self, id, input_dim, timesteps,
                 n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1],
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans', save_dir='results/tmp'):
        assert (timesteps % pool_size == 0)
        self.model_name = "ldtc_" + str(id)
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None
        self.res = []
        self.save_dir = save_dir
        self.cluster_centers_old = None

    def initialize(self, n_clusters):

        self.autoencoder, self.encoder, self.decoder = temporal_autoencoder_v2(input_dim=self.input_dim,
                                                                               timesteps=self.timesteps,
                                                                               n_filters=self.n_filters,
                                                                               kernel_size=self.kernel_size,
                                                                               pool_size=self.pool_size,
                                                                               n_units=self.n_units)
        self.n_clusters = n_clusters
        self.stable_pc = np.zeros((n_clusters, 1))
        clustering_layer = TCLayer(self.n_clusters,
                                   alpha=self.alpha,
                                   dist_metric=self.dist_metric,
                                   name='TCLayer')(self.encoder.output)

        self.model = Model(inputs=self.autoencoder.input,
                           outputs=clustering_layer)

    # Reconstruct the clustering layer TCLayer
    def fix_model(self, n_clusters, optimizer='adam'):
        self.cluster_centers_old = self.cluster_centers_
        self.n_clusters = n_clusters + self.n_clusters
        clustering_layer = TCLayer(self.n_clusters,
                                   alpha=self.alpha,
                                   dist_metric=self.dist_metric,
                                   name='TCLayer')(self.encoder.output)

        self.model = Model(inputs=self.autoencoder.input,
                           outputs=clustering_layer)
        self.model.compile(loss='kld', optimizer=optimizer)

    @property
    def cluster_centers_(self):

        return self.model.get_layer(name='TCLayer').get_weights()[0]

    def compile(self, pretrain_optimizer='adam', optimizer='adam'):

        self.autoencoder.compile(loss='mse', optimizer=pretrain_optimizer)
        self.model.compile(loss='kld', optimizer=optimizer)

    def load_weights(self, weights_path):

        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):

        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):

        if self.dist_metric == 'eucl':
            return tsdistances.eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return tsdistances.cid(x1, x2)
        elif self.dist_metric == 'cor':
            return tsdistances.cor(x1, x2)
        elif self.dist_metric == 'acf':
            return tsdistances.acf(x1, x2)
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')

    def init_cluster_weights(self, X, is_fine_tuning=False):

        assert (self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)
        if is_fine_tuning:
            km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])
            self.model.get_layer(name='TCLayer').set_weights([cluster_centers])
            labels = km.labels_
        else:
            if self.cluster_init == 'hierarchical':
                if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                    hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                                 affinity='euclidean',
                                                 linkage='complete').fit(features.reshape(features.shape[0], -1))
                else:  # compute distance matrix using dist
                    d = np.zeros((features.shape[0], features.shape[0]))
                    for i in range(features.shape[0]):
                        for j in range(i):
                            d[i, j] = d[j, i] = self.dist(features[i], features[j])
                    hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                                 affinity='precomputed',
                                                 linkage='complete').fit(d)
                # compute centroid
                cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
                labels = hc.labels_
            elif self.cluster_init == 'kmeans':
                # fit k-means on flattened features
                km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features.reshape(features.shape[0], -1))
                cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])
                labels = km.labels_

            if self.cluster_centers_old is not None:
                cluster_centers = np.concatenate((self.cluster_centers_old, cluster_centers), axis=0)
                self.model.get_layer(name='TCLayer').set_weights([cluster_centers])
            else:
                self.model.get_layer(name='TCLayer').set_weights([cluster_centers])
        print('Done!')
        return labels

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def pretrain(self, X, epochs=10, batch_size=64, save_dir='results/snapshot', verbose=1):

        print('Pretraining...')
        # if X_snapshot is not None:
        #     X_concat = np.concatenate((X, X_snapshot), axis=0)
        #     indexs = random.sample(list(range(X_concat.shape[0])), int(X_concat.shape[0] * 0.05))
        #     sample_partial = X_concat[indexs::]
        #     np.save(save_dir + '/' + self.model_name + '_snapshot.npy', sample_partial)
        # else:
        #     X_concat = X

        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=verbose)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True

    def mixture_replay(self, X_train, count):
        y_pred = self.predict(X_train)
        num = np.bincount(y_pred).min()
        data = data_cluster_divide(X_train, y_pred)
        X_replay_ = []
        y_replay_ = []

        index = random.sample(range(num), count)

        for i in range(len(data)):
            X_ = data[i][0][index]
            y_ = data[i][1][index]
            X_replay_.append(X_)
            y_replay_.append(y_)

        return np.array(X_replay_), np.array(y_replay_)

    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            epochs=100,
            eval_epochs=10,
            save_epochs=10,
            batch_size=64,
            tol=0.001,
            patience=5):

        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        # Logging file
        logfile = open(self.save_dir + '/ldtc_log.csv', 'w')
        fieldnames = ['epoch', 'T', 'L', 'Lr', 'Lc']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lc_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        y_pred_last = None
        patience_cnt = 0

        print(
            'Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs,
                                                                                                    save_epochs))
        res = 0
        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train)
            p = LDTC.target_distribution(q)
            mean_p = p.mean(axis=1)
            self.stable_pc = (mean_p * 0.1 + self.stable_pc * 0.9) / (1 - pow(0.9, epoch))

            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                # Initialize log dictionary
                logdict = dict(epoch=epoch)

                y_pred = q.argmax(axis=1)
                if X_val is not None:
                    q_val = self.model.predict(X_val)
                    p_val = LDTC.target_distribution(q_val)
                    y_val_pred = q_val.argmax(axis=1)

                print('epoch {}'.format(epoch))

                loss = self.model.evaluate(X_train, p, batch_size=batch_size, verbose=False)
                print('[Train] - total loss={:f}'.format(loss))

                if X_val is not None:
                    val_loss = self.model.evaluate(X_val, [X_val, p_val], batch_size=batch_size, verbose=False)
                    print('[Val] - total loss={:f}'.format(val_loss))

                # Evaluate the clustering performance using labels
                if y_train is not None:
                    # print("np.unique(y_train):", np.unique(y_pred))
                    logdict['acc'] = cluster_acc(y_train, y_pred)
                    logdict['pur'] = cluster_purity(y_train, y_pred)
                    logdict['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
                    logdict['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
                    print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'],
                                                                                    logdict['nmi'], logdict['ari']))

                    if (logdict['acc'] + logdict['pur']) > res:
                        res = logdict['acc'] + logdict['pur']
                        acc_best = logdict['acc']
                        pur_best = logdict['pur']
                        model_best = clone_model(self.model)
                        model_best.set_weights(self.model.get_weights())

                if y_val is not None:
                    logdict['acc_val'] = cluster_acc(y_val, y_val_pred)
                    logdict['pur_val'] = cluster_purity(y_val, y_val_pred)
                    logdict['nmi_val'] = metrics.normalized_mutual_info_score(y_val, y_val_pred)
                    logdict['ari_val'] = metrics.adjusted_rand_score(y_val, y_val_pred)
                    print(
                        '[Val] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc_val'], logdict['pur_val'],
                                                                                logdict['nmi_val'], logdict['ari_val']))

                logwriter.writerow(logdict)

                # check stop criterion
                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and assignment_changes < tol:
                    patience_cnt += 1
                    print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes,
                                                                                                    tol, patience_cnt,
                                                                                                    patience))
                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        logfile.close()
                        break
                else:
                    patience_cnt = 0

            # Save intermediate model and plots
            if epoch % save_epochs == 0:
                self.model.save_weights(self.save_dir + '/LDTC_model_' + str(epoch) + '.h5')
                print('Saved model to:', self.save_dir + '/LDTC_model_' + str(epoch) + '.h5')

            # Train for one epoch
            self.model.fit(X_train, p, epochs=1, batch_size=batch_size, verbose=False)

        self.res.append([acc_best, pur_best])
        # Save the final model
        logfile.close()
        print('Saving model to:', self.save_dir + '/LDTC_model_final.h5')
        self.model.save_weights(self.save_dir + '/LDTC_model_final.h5')
        self.model = clone_model(model_best)
        self.model.set_weights(model_best.get_weights())
        self.X_snapshot, self.y_snapshot = self.mixture_replay(X_train, int(X_train.shape[0] * 0.05))

    def get_gap_before_pretrain(self, X):
        z = self.encoder.predict(X, verbose=0)
        q = self.model.predict(X, verbose=0)
        p = LDTC.target_distribution(q)
        mean_p = p.mean(axis=1)
        gap = self.dist(self.stable_pc, mean_p)

        return gap

    # Candidate function. This function is not used now.
    # def interpolation_training(self, X_train, y_train, data_inter_count=10000, epochs=30, batch_size=64,
    #                            last_optimizer='adam'):
    #
    #     print('Interpolation training for clustering layer.')
    #     t0 = time()
    #
    #     z = self.encode(X_train)
    #     y = self.model.predict(X_train, verbose=0)
    #
    #     interpolation_z_train, interpolation_y_train = mixup_data(z, y, data_inter_count)
    #
    #     clustering_input = Input(shape=(z.shape[1], z.shape[2]), name='cluster_input')
    #     clustering_layer = TCLayer(self.n_clusters,
    #                                alpha=self.alpha,
    #                                dist_metric=self.dist_metric,
    #                                name='TCLayer')(clustering_input)
    #     inter_train_clustering_model = Model(inputs=clustering_input, outputs=clustering_layer)
    #     inter_train_clustering_model.compile(loss='kld', optimizer=last_optimizer)
    #     inter_train_clustering_model.get_layer(name='TCLayer').set_weights(
    #         [self.model.get_layer(name='TCLayer').get_weights()[0]])
    #
    #     for epoch in range(epochs):
    #         # Train for one epoch
    #         inter_train_clustering_model.fit(interpolation_z_train, interpolation_y_train, epochs=1,
    #                                          batch_size=batch_size, verbose=False)
    #
    #         q = self.model.predict(X_train)
    #         y_pred = q.argmax(axis=1)
    #         print('epoch {}'.format(epoch))
    #         cluster_results = cluster_evaluation(y_train, y_pred)
    #         print('[Interpolation Training] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(cluster_results['acc'],
    #                                                                                          cluster_results['pur'],
    #                                                                                          cluster_results['nmi'],
    #                                                                                          cluster_results['ari']))
    #
    #     self.model.get_layer(name='TCLayer').set_weights(
    #         [inter_train_clustering_model.get_layer(name='TCLayer').get_weights()[0]])
    #
    #     print()
    #     print('Interpolation training for decoder.')
    #     for layer in self.encoder.layers:
    #         layer.trainable = False
    #         print('Layer Name: ', layer.name, ' trainable: ', layer.trainable)
    #
    #     self.encoder.trainable = False
    #
    #     for epoch in range(epochs):
    #         self.autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=1, verbose=1)
    #         loss = self.autoencoder.evaluate(X_train, X_train, batch_size=batch_size, verbose=False)
    #
    #     for layer in self.encoder.layers:
    #         layer.trainable = True
    #         print('Layer Name: ', layer.name, ' trainable: ', layer.trainable)
    #     self.encoder.trainable = True
    #
    #     print('Interpolation training time: ', time() - t0)

    # Save the model
    def model_save(self):
        self.decoder.save_weights(self.save_dir + '/TSC_decoder_' + self.model_name + '.h5')
        self.encoder.save_weights(self.save_dir + '/TSC_encoder_' + self.model_name + '.h5')
        self.model.save_weights(self.save_dir + '/TSC_model_' + self.model_name + '.h5')
