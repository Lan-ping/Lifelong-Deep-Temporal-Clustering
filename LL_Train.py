# Utilities
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings('ignore')
import argparse
from time import time
from data_util.metrics import *
from model import LDTC


class ModelManager:

    def __init__(self, input_dim, timesteps,
                 n_filters=50, kernel_size=10, pool_size=10, n_units=[50, 1],
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans',
                 pretrain_optimizer='adam', optimizer='adam', save_dir='results/tmp', model_capacity=10,
                 ll_threshold=0.05):

        assert (timesteps % pool_size == 0)
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.pretrain_optimizer = pretrain_optimizer
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.model_counter = 0
        self.model_capacity = model_capacity

        self.models = []
        self.models_usage_num = []
        self.ll_threshold = ll_threshold

    def init_new_model(self, n_clusters):
        self.model_counter = self.model_counter + 1
        dtc = LDTC(self.model_counter, input_dim=self.input_dim, timesteps=self.timesteps,
                   n_filters=self.n_filters, kernel_size=self.kernel_size,
                   pool_size=self.pool_size, n_units=self.n_units, alpha=self.alpha, dist_metric=self.dist_metric,
                   cluster_init=self.cluster_init, save_dir=self.save_dir)
        dtc.initialize(n_clusters)
        dtc.model.summary()
        dtc.decoder.summary()
        dtc.compile(pretrain_optimizer=self.pretrain_optimizer, optimizer=self.optimizer)
        return dtc

    def get_similar_model_or_initialize(self, X, n_clusters):
        # Calculate each existing model's score for new data
        scores = list(map(lambda model: model.get_gap_before_pretrain(X), self.models))
        # Filter out models with scores less than the threshold
        satisfied = list(filter(lambda score: score < self.ll_threshold, scores))

        if len(satisfied) != 0:  # There are models that satisfy the conditions
            is_fine_tuning = True
            index = scores.index(min(scores))
            dtc = self.models[index]
            self.models_usage_num[index] = self.models_usage_num[index] + 1
            dtc.fix_model(n_clusters, optimizer=self.optimizer)
        else:  # There are no models that meet the conditions.
            is_fine_tuning = False
            dtc = self.init_new_model(n_clusters)

        return dtc, is_fine_tuning

    def model_train(self, X_train, y_train, n_clusters, pretrain_epochs=10, epochs=100, eval_epochs=10, save_epochs=10,
                    batch_size=64, tol=0.001, patience=5, is_interpolation_training=False):

        dtc, is_fine_tuning = self.get_similar_model_or_initialize(X_train, n_clusters)

        if not is_fine_tuning:
            # Pre-train phase
            dtc.pretrain(X=X_train, epochs=pretrain_epochs, batch_size=batch_size, save_dir=self.save_dir)
            # Initialize cluster centroids
            dtc.init_cluster_weights(X_train)
            # Train phase
            t0 = time()
            dtc.fit(X_train=X_train, y_train=y_train, epochs=epochs, eval_epochs=eval_epochs, save_epochs=save_epochs,
                    batch_size=batch_size, tol=tol, patience=patience)
            print('Training time: ', (time() - t0))

        else:
            # Initialize cluster centroids
            cluster_pred = dtc.init_cluster_weights(X_train, is_fine_tuning)
            cluster_results = cluster_evaluation(y_train, cluster_pred)
            print("The result of init clustering")
            print(cluster_results)

            # Train phase
            t0 = time()
            X_concat = np.concatenate((X_train, dtc.X_snapshot), axis=0)
            y_concat = np.concatenate((y_train, dtc.y_snapshot), axis=0)
            dtc.fit(X_train=X_concat, y_train=y_concat, epochs=epochs, eval_epochs=eval_epochs, save_epochs=save_epochs,
                    batch_size=batch_size, tol=tol, patience=patience)
            print('Training time: ', (time() - t0))

        # Interpolation training
        # if is_interpolation_training:
        #     dtc.interpolation_training(X_train, y_train)

        # Evaluate
        if not is_fine_tuning:
            describe = "Not fine_tuning"
        else:
            describe = "Fine_tuning"

        print('Performance (TRAIN) ****** {}'.format(describe))
        q = dtc.model.predict(X_train)
        y_pred = q.argmax(axis=1)
        cluster_results = cluster_evaluation(y_train, y_pred)
        print("The result of fit clustering")
        print(cluster_results)

        # Save model in RAM or Disk
        if not is_fine_tuning and len(self.models) >= self.model_capacity:
            min_usage = min(self.models_usage_num)
            index = self.models_usage_num.index(min_usage)
            self.models[index].model_save()
            self.models[index] = dtc
            self.models_usage_num[index] = 1
        elif not is_fine_tuning and len(self.models) < self.model_capacity:
            self.models.append(dtc)
            self.models_usage_num.append(1)


if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='data', required=True,
                        help='the folder path where the temporal data can be read. \n There can be many pieces of data, but the shape of the data must be consistent.\n The data file names must start with \'X_train\' and \'y_train\', are similar to X_train_1.npy and y_train_1.npy, X_train_2.npy and y_train_2.npy')
    parser.add_argument('--timesteps', default=100, type=int, required=True, help='time series length')
    parser.add_argument('--input_dim', default=100, type=int, required=True, help='number of variables')
    # parser.add_argument('--n_clusters', default='3,4', type=str,
    #                     help='Number of clusters in advance for each dataset. e.g. \"3,4,5\"')
    parser.add_argument('--n_filters', default=50, type=int, help='number of filters in convolutional layer')
    parser.add_argument('--kernel_size', default=10, type=int, help='size of kernel in convolutional layer')
    parser.add_argument('--pool_size', default=3, required=True, type=int, help='pooling size in max pooling layer')
    parser.add_argument('--n_units', nargs=2, default=[50, 1], type=int, help='numbers of units in the BiLSTM layers')
    parser.add_argument('--alpha', default=1.0, type=float, help='coefficient in Student\'s kernel')
    parser.add_argument('--dist_metric', default='eucl', type=str, choices=['eucl', 'cid', 'cor', 'acf'],
                        help='distance metric between latent sequences')
    parser.add_argument('--cluster_init', default='kmeans', type=str, choices=['kmeans', 'hierarchical'],
                        help='cluster initialization method')
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--tol', default=0.001, type=float, help='tolerance for stopping criterion')
    parser.add_argument('--patience', default=5, type=int, help='patience for stopping criterion')
    parser.add_argument('--save_dir', default='results/tmp')
    parser.add_argument('--model_capacity', default=10, type=int, help='model capacity')
    parser.add_argument('--ll_threshold', default=0.05, type=float, help='Lifelong learning threshold')
    args = parser.parse_args()
    print(args)

    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set default values
    pretrain_optimizer = 'adam'
    optimizer = 'adam'
    mm = ModelManager(input_dim=args.input_dim,
                      timesteps=args.timesteps,
                      n_filters=args.n_filters,
                      kernel_size=args.kernel_size,
                      pool_size=args.pool_size,
                      n_units=args.n_units,
                      alpha=args.alpha,
                      dist_metric=args.dist_metric,
                      cluster_init=args.cluster_init,
                      pretrain_optimizer=pretrain_optimizer,
                      optimizer=optimizer,
                      save_dir=args.save_dir,
                      model_capacity=args.model_capacity,
                      ll_threshold=args.ll_threshold)

    # Load data
    data_train_name_list = list(filter(lambda fileName: fileName.startswith('X_train'), os.listdir(args.data_path)))
    for i in range(len(data_train_name_list)):
        X_val = None
        y_val = None
        X_train_path = args.data_path + '/X_train_{}.npy'.format(i + 1)
        y_train_path = args.data_path + '/y_train_{}.npy'.format(i + 1)
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)

        print("Lifelong Temporal Clustering for data: X_train_{}".format(i + 1))

        n_clusters = len(np.unique(y_train))

        t0 = time()
        # Select and train the LDTC model
        mm.model_train(X_train=X_train,
                       y_train=y_train,
                       n_clusters=n_clusters,
                       pretrain_epochs=args.pretrain_epochs,
                       epochs=args.epochs,
                       eval_epochs=args.eval_epochs,
                       save_epochs=args.save_epochs,
                       batch_size=args.batch_size,
                       tol=args.tol,
                       patience=args.patience)

        print('Running time: ', (time() - t0))
        print()
        print()
