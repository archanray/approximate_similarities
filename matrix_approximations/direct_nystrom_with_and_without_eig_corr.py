import numpy as np
import matplotlib.pyplot as plt
import sys
# import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_mat_file

from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat
import random
import os

from plotter import plot_errors
from recursiveNystrom import wrapper_for_recNystrom

def nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error", correct_outer=False):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        if correct_outer == False:
            similarity_matrix_x = deepcopy(similarity_matrix)
        else:
            similarity_matrix_x = deepcopy(similarity_matrix)\
                                  - min_eig*np.eye(len(similarity_matrix))
    else:
        similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def CUR_alt(similarity_matrix, k, mult=2, return_type="error"):
    """
    compute CUR approximation
    versions:
    U = S2^T K S1
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices_rows = np.sort(random.sample(\
                     list_of_available_indices, min(k, len(similarity_matrix))))
    if mult > 1:
        sample_indices_cols = np.sort(random.sample(\
                         list(sample_indices_rows), int(k/mult)))
    else:
        sample_indices_cols = np.sort(random.sample(\
                     list_of_available_indices, min(k, len(similarity_matrix))))
    A = similarity_matrix[sample_indices_rows][:, sample_indices_cols]
    
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS_cols = similarity_matrix_x[:, sample_indices_cols]
    KS_rows = similarity_matrix_x[sample_indices_rows]
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS_cols @ np.linalg.pinv(A) @ KS_rows)\
                / np.linalg.norm(similarity_matrix)


def ratio_nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error"):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)
    elif min_eig_mode == False:
        similarity_matrix_x = deepcopy(similarity_matrix)
    else:
        local_min_eig = min(0, np.min(np.linalg.eigvals(A))) - eps
        ratio = min_eig / local_min_eig
        A = (1.0/ratio)*A - np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)

    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        if min_eig_mode == True or min_eig_mode == False:
            return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)
        else:
            return np.linalg.norm(\
                similarity_matrix - \
                (1/ratio)*KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", \
    scaling=False, mult=0, eig_mult=1):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k*len(similarity_matrix)))
    else:
        large_k = min(mult*k, len(similarity_matrix)-1)
    larger_sample_indices = np.sort(random.sample(\
                            list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]
    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    min_eig = eig_mult*min_eig
    if scaling == True:
        min_eig = min_eig*np.float(len(similarity_matrix))/np.float(large_k)


    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig


def nystrom_with_eig_estimate_sub(similarity_matrix, k, return_type="error", \
    scaling=False, mult=0, eig_mult=1, new_rescale=False):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    # sample_indices = np.sort(random.sample(\
    #                  list_of_available_indices, k))
    # A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k*len(similarity_matrix)))
    else:
        large_k = min(mult*k, len(similarity_matrix)-1)
    larger_sample_indices = np.sort(random.sample(\
                            list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]

    sample_indices = np.sort(random.sample(\
                            list(larger_sample_indices), k))
    A = similarity_matrix[sample_indices][:, sample_indices]

    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    min_eig = eig_mult*min_eig
    if scaling == True:
        min_eig = min_eig*np.float(len(similarity_matrix))/np.float(large_k)

    if new_rescale:
        alpha = np.linalg.norm(A) / (np.linalg.norm(A - min_eig*np.eye(len(A))))
    else:
        alpha = 1
    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A)/alpha @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig


def CUR(similarity_matrix, k, eps=1e-3, delta=1e-14, return_type="error", same=False):
    """
    implementation of Linear time CUR algorithm of Drineas2006 et. al.

    input:
    1. similarity matrix in R^{n,d}
    2. integers c, r, and k

    output:
    1. either C, U, R matrices
    or
    1. CU^+R
    or
    1. error = similarity matrix - CU^+R

    """
    n,d = similarity_matrix.shape
    # setting up c, r, eps, and delta for error bound
    # c = (64*k*((1+8*np.log(1/delta))**2) / (eps**4)) + 1
    # r = (4*k / ((delta*eps)**2)) + 1
    # c = 4*k
    c = k
    r = k
    if c > n:
        c= n
    # r = 4*k
    if r > n:
        r = n
    # print("chosen c, r:", c,r)
    try:
        assert 1 <= c and c <= d
    except AssertionError as error:
        print("1 <= c <= m is not true")
    try:
        assert 1 <= r and r <= n
    except AssertionError as error:
        print("1 <= r <= n is not true")
    try:
        assert 1 <= k and k <= min(c,r)
    except AssertionError as error:
        print("1 <= k <= min(c,r)")

    # using uniform probability instead of row norms
    pj = np.ones(d).astype(float) / float(d)
    qi = np.ones(n).astype(float) / float(n)

    # choose samples
    samples_c = np.random.choice(range(d), c, replace=False, p = pj)
    if same:
        samples_r = samples_c
    else:
        samples_r = np.random.choice(range(n), r, replace=False, p = qi)

    # grab rows and columns and scale with respective probability
    samp_pj = pj[samples_c]
    samp_qi = qi[samples_r]
    C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj*c)
    rank_k_C = C
    # modification works only because we assume similarity matrix is symmetric
    R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi*r)
    R = R.T
    psi = C[samples_r, :].T / np.sqrt(samp_qi*r)
    psi = psi.T

    U = np.linalg.pinv(rank_k_C.T @ rank_k_C)
    # i chose not to compute rank k reduction of U
    U = U @ psi.T
    
    if return_type == "decomposed":
        return C, U, R
    if return_type == "approximate":
        return (C @ U) @ R
    if return_type == "error":
        # print(np.linalg.norm((C @ U) @ R))
        relative_error = np.linalg.norm(similarity_matrix - ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
        return relative_error

def Lev_samples(similarity_matrix, k, KS_correction=False):
    K = similarity_matrix
    num_imp_samples = k
    error, _,_,_ = wrapper_for_recNystrom(similarity_matrix, K, num_imp_samples, \
        runs=1, mode="normal", normalize="rows", \
        expand=True, KS_correction=KS_correction)
    return error

step = 50
runs_ = 3
"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filetype = None
dataset = sys.argv[1]
if dataset == "PSD":
    feats = np.random.random((1000,1000))
    similarity_matrix = feats @ feats.T
    filetype = "numpy"
if dataset == "mrpc" or dataset == "rte" or dataset == "stsb":
    filename = "../GYPSUM/"+dataset+"_predicts_0.npy"
    filetype = "python"
if dataset == "twitter":
    similarity_matrix = read_mat_file(file_="./WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
if dataset == "ohsumed":
    similarity_matrix = read_mat_file(file_="./WordMoversEmbeddings/mat_files/oshumed_K_set1.mat", version="v7.3")
if dataset == "recipe":
    similarity_matrix = read_mat_file(file_="/mnt/nfs/work1/elm/ray/recipe_trainData.mat", version="v7.3")
if dataset == "news":
    similarity_matrix = read_mat_file(file_="/mnt/nfs/work1/elm/ray/20ng2_new_K_set1.mat", version="v7.3")
if filetype == "python":
    similarity_matrix = read_file(filename)
# similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")

true_error = []
KS_corrected_error_list = []
KS_ncorrected_error_list = []

scaling_error_list = []
nscaling_error_list = []

min_eig_scaling = []
min_eig_nscaling = []

SKS_corrected_error_list = []
SKS_ncorrected_error_list = []
SKS_rcorrected_error_list = []
ZKZ_multiplier_error_list = []

CUR_diff_error_list = []
CUR_same_error_list = []
CUR_alt_error_list = []

Lev_corrected_error_list = []
Lev_ncorrected_error_list = []

# check for similar rows or columns
if dataset != "PSD":
    unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
    similarity_matrix_O = similarity_matrix[indices][:, indices]
    # symmetrization
    similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
    # print("is the current matrix PSD? ", is_pos_def(similarity_matrix))
id_count = len(similarity_matrix)-1
print(dataset)

# if filename == "rte":
#   similarity_matrix = 1-similarity_matrix
#   similarity_matrix_O = 1-similarity_matrix_O

################# uniform sampling #####################################
print("original variations of nystrom")
eps=1e-16
min_eig = np.min(np.linalg.eigvals(similarity_matrix)) - eps
for k in tqdm(range(10, id_count, 10)):
    # err = 0
    # for j in range(runs_):
    #     error = nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # KS_ncorrected_error_list.append(error)
    # pass

    err = 0
    for j in range(runs_):
        error = nystrom(similarity_matrix, k, min_eig_mode=False)
        err += error
    error = err/np.float(runs_)
    true_error.append(error)
    pass    

######################## eigen corrected uniform sampling ###################
# eps=1e-16
# min_eig_val = min(0, np.min(np.linalg.eigvals(similarity_matrix))) - eps
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig_val)
#         err += error
#     error = err/np.float(runs_)
#     KS_ncorrected_error_list.append(error)
#     pass


# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     min_eig_agg = 0
#     for j in range(runs_):
#         error, min_eig = nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", scaling=True)
#         err += error
#         min_eig_agg += min_eig
#     error = err/np.float(runs_)SS
#     scaling_error_list.append(error)
#     pass

print("our Nystrom")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    min_eig_agg = 0
    for j in range(runs_):
        error, min_eig = nystrom_with_eig_estimate_sub(similarity_matrix, k, \
            return_type="error", mult=2, eig_mult=1.5, new_rescale=False)
        err += error
        min_eig_agg += min_eig
    error = err/np.float(runs_)
    nscaling_error_list.append(error)
    pass    

# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     min_eig_agg = 0
#     for j in range(runs_):
#         error, min_eig = nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", mult=10)
#         err += error
#         min_eig_agg += min_eig
#     error = err/np.float(runs_)
#     ZKZ_multiplier_error_list.append(error)
#     pass   

################################## RATIO CHECK ################################
# eps=1e-16
# min_eig = min(0, np.min(np.linalg.eigvals(similarity_matrix))) - eps
# for k in tqdm(range(10, id_count, 10)):
    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_corrected_error_list.append(error)
    # pass

    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode=False, min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_ncorrected_error_list.append(error)
    # pass    

    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode="ratio", min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_rcorrected_error_list.append(error)
    # pass 

################################ CUR decomposition ##########################
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = CUR(similarity_matrix, k)
#         err += error
#     error = err/np.float(runs_)
#     CUR_diff_error_list.append(error)
#     pass

print("CUR")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    for j in range(runs_):
        error = CUR(similarity_matrix, k, same=True)
        err += error
    error = err/np.float(runs_)
    CUR_same_error_list.append(error)
    pass

print("CUR diff")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    for j in range(runs_):
        error = CUR(similarity_matrix, k, same=False)
        err += error
    error = err/np.float(runs_)
    CUR_diff_error_list.append(error)
    pass

SiCUR_error_list = []
print("SiCUR")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    for j in range(runs_):
        error = CUR_alt(similarity_matrix, k)
        err += error
    error = err/np.float(runs_)
    SiCUR_error_list.append(error)
    pass

SkCUR_error_list = []
print("Skeleton approx")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    for j in range(runs_):
        error = CUR_alt(similarity_matrix, k, mult=1)
        err += error
    error = err/np.float(runs_)
    SkCUR_error_list.append(error)
    pass


# print(CUR_same_error_list)
################################ leverage scores #############################
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = Lev_samples(similarity_matrix, k)
#         err += error
#     error = err/np.float(runs_)
#     Lev_ncorrected_error_list.append(error)
#     pass

# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = Lev_samples(similarity_matrix, k, KS_correction=True)
#         err += error
#     error = err/np.float(runs_)
#     Lev_corrected_error_list.append(error)
#     # Lev_error_list.append(error)
#     pass

#######################################################################
# SAVE
import pickle
with open("post_exp/"+dataset+"_nystrom_only_vals.pkl", "wb") as f:
    pickle.dump([true_error, \
        nscaling_error_list, \
        CUR_same_error_list, \
        CUR_diff_error_list, \
        SiCUR_error_list, \
        SkCUR_error_list], f)
# plt.plot(SiCUR_alt_error_list, label="CUR alt")
# plt.plot(true_error, label="nystrom")
# plt.plot(CUR_diff_error_list, label="CUR diff")
# plt.plot(CUR_same_error_list, label="CUR same")
# plt.plot(SkCUR_alt_error_list, label="CUR alt")
# plt.legend(loc="upper right")
# # plt.ylim(top=2.0, bottom=0.0)
# plt.show()
#######################################################################
# # PLOTS
# plot_errors([nscaling_error_list, CUR_same_error_list], \
#     id_count= id_count, \
#     labels=["Nystrom", "CUR-same"], \
#     name="Twitter",\
#     save_path="comparison_with_CUR")#, y_lims=[0,1])