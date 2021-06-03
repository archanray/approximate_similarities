from liblinear.liblinearutil import *
import numpy as np
from matplotlib import pyplot as plt
# import torch
import sys
from approximator import nystrom_with_eig_estimate as nystrom
from approximator import nystrom_with_samples as nystrom_test
from approximator import CUR
from approximator import CUR_with_samples as CUR_test
from approximator import CUR_alt
from approximator import CUR_alt_with_samples as CUR_alt_test
from utils import read_mat_file
from scipy.linalg import  sqrtm
from scipy.linalg import svd
from sklearn.model_selection import StratifiedKFold
# import torch
# import torch.nn as nn
# import torch.optim as optim

from absl import flags
from absl import logging
from absl import app
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', "twitter", "twitter or ohsumed or news or recipe")
flags.DEFINE_string('method', "nystrom", "method for approximation")
flags.DEFINE_float('lambda_inverse', 1e4, "lambda inverse value")
flags.DEFINE_float('gamma', 0.1, "exp(-dsr/gamma)")
flags.DEFINE_integer('sample_size', 500, "number of samples to be considered")
flags.DEFINE_float('svme', 0.0001, "e paprameter of svm")
flags.DEFINE_string('mode', "train", "run mode")

logging.set_verbosity(logging.INFO)

def get_feat(X, indices, k, gamma, approximator, mode="train", \
    samples=None, min_eig=None, SKS_prev=None, U_prev=None, S_prev=None, row_samples=None, col_samples=None):
    eps=1e-3
    if approximator == "nystrom":
        if mode == "train":
            KS, SKS, sample_indices, min_eig = nystrom(X, k, \
                                  return_type="decomposed", \
                                  mult=2, eig_mult=1.5, indices=list(indices), \
                                  gamma=gamma)
            feats = KS @ sqrtm(np.linalg.inv(SKS))
            return feats, sample_indices, min_eig, SKS
        if mode == "val":
            KS, SKS = nystrom_test(X, indices, samples, min_eig, gamma=gamma)
            if SKS_prev is None:
                feats = KS @ sqrtm(np.linalg.inv(SKS))
            else:
                feats = KS @ sqrtm(np.linalg.inv(SKS_prev))
            return feats
    
    if approximator == "CUR_alt":
        if mode == "train":
            KS1, S2KS1, row_indices, col_indices = CUR_alt(X, k, \
                                  return_type="decomposed", \
                                  indices=list(indices), \
                                  gamma=gamma)
            chosen_rows = []
            for i in range(len(row_indices)):
                if row_indices[i] in col_indices:
                    chosen_rows.append(i)
            for i in range(S2KS1.shape[1]):
                S2KS1[i,i] = S2KS1[chosen_rows[i],i]+eps
            u,s, vh = svd(S2KS1, full_matrices=True)
            feats = KS1 @ (vh @ sqrtm(np.linalg.pinv(np.diag(s))))
            return feats, row_indices, col_indices, vh @ sqrtm(np.linalg.pinv(np.diag(s)))
        if mode == "val":
            KS1 = CUR_alt_test(X, indices, col_samples, gamma=gamma)
            feats = KS1 @ S_prev
            return feats


    if approximator == "CUR":
        if mode == "train":
            C, U, sample_indices = CUR(X, k,\
                                       indices=list(indices),\
                                       return_type="decomposed",\
                                       gamma=gamma)
            # avoid U being singular
            U = U+eps*np.eye(U.shape[0])
            u, s, vh = svd(np.linalg.inv(U), full_matrices=True)
            feats = C @ (u @ sqrtm(np.diag(s)))
            
            return feats, sample_indices, u, s
        if mode == "val":
            C, U = CUR_test(X, indices, samples=samples, gamma=gamma)
            if U_prev is None:
                # avoid U being singular
                U = U+eps*np.eye(U.shape[0])
                u, s, vh = svd(np.linalg.inv(U), full_matrices=True)
            else:
                u = U_prev
                s = S_prev
            feats = C @ (u @ sqrtm(np.diag(s)))
            return feats


# training
def train_all(X, Y, config):
    # create train and validation splits
    kf = StratifiedKFold(n_splits=config["CV"], random_state=None, shuffle=True)

    valAccu = []
    for train_index, val_index in kf.split(X, Y):
        Y_train, Y_val = Y[train_index], Y[val_index]

        # train features
        if config["approximator"] == "nystrom":
            train_feats, indices, eig, SKS = \
                            get_feat(X, train_index, \
                                config["samples"], \
                                config["gamma"], \
                                config["approximator"])

            # validation features
            val_feats = get_feat(X, val_index, \
                                  config["samples"], \
                                  config["gamma"], \
                                  config["approximator"], \
                                  mode="val", \
                                  samples=indices, \
                                  min_eig=eig, \
                                  SKS_prev=SKS)
        if config["approximator"] == "CUR":
            train_feats, indices, U_train, S_train = \
                            get_feat(X, train_index, \
                                config["samples"], \
                                config["gamma"], \
                                config["approximator"])

            # validation features
            val_feats = get_feat(X, val_index, \
                                  config["samples"], \
                                  config["gamma"], \
                                  config["approximator"], \
                                  mode="val", \
                                  samples=indices, \
                                  U_prev=U_train,\
                                  S_prev=S_train)

        if config["approximator"] == "CUR_alt":
            train_feats, row_indices, col_indices, S_train = \
                            get_feat(X, train_index, \
                                config["samples"], \
                                config["gamma"], \
                                config["approximator"])

            # validation features
            val_feats = get_feat(X, val_index, \
                                  config["samples"], \
                                  config["gamma"], \
                                  config["approximator"], \
                                  mode="val", \
                                  S_prev=S_train,\
                                  row_samples = row_indices,\
                                  col_samples = col_indices)

        # hyperparameters
        s = "-s 2 -e "+str(config["e"])+" -n 8 -q -c "+str(config["lambda_inverse"])
        # train model
        model_linear = train(Y_train, train_feats, s)
        # predict on validation
        [_, val_accuracy, _] = predict(Y_val, val_feats, model_linear)
        # validation accuracy
        valAccu.append(val_accuracy[0])
    print(np.mean(valAccu), np.std(valAccu))
    wandb.log({"validation_mean":np.mean(valAccu), \
       "validation_std":np.std(valAccu)})
    logging.info("validation_accuracy: %s", np.mean(valAccu))
    return None

# main
def main(argv):
    wandb.init(project="WME-Nyst and CUR")
    wandb.config.update(flags.FLAGS)
    logging.info('Running with args %s', str(argv))

    # get dataset
    dataset = FLAGS.dataset
    if dataset == "ohsumed":
        filename = "oshumed_K_set1.mat"
        version = "v7.3"
    if dataset == "twitter":
        filename = "twitter_K_set1.mat"
        version = "default"
    if dataset == "news":
        filename = "20ng2_new_K_set1.mat"
        version = "v7.3"
    if dataset == "recipe":
        filename = "recipe_trainData.mat"
        version = "v7.3"

    approximator = FLAGS.method
    if approximator not in ["nystrom", "CUR", "CUR_alt"]:
        print("please choose among nystrom, CUR_alt and CUR for approximator")
        return None

    # get EMD matrix
    similarity_matrix, labels = read_mat_file(\
                                    file_="./"+filename,\
                                    version=version, return_type="all")

    # set hyperparameters
    config = {"samples":FLAGS.sample_size,\
              "CV":10, \
              "e":FLAGS.svme, \
              "gamma": FLAGS.gamma,\
              "lambda_inverse":FLAGS.lambda_inverse,\
              "approximator":approximator}

    train_all(similarity_matrix, labels, config)
    return None

if __name__ == "__main__":
    print(FLAGS)
    app.run(main)
