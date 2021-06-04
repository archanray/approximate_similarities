import numpy as np
import sys
from tqdm import tqdm

from copy import deepcopy
import random

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
    sample_indices = np.sort(random.sample( \
        list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig * np.eye(len(A))
        if correct_outer == False:
            similarity_matrix_x = deepcopy(similarity_matrix)
        else:
            similarity_matrix_x = deepcopy(similarity_matrix) \
                                  - min_eig * np.eye(len(similarity_matrix))
    else:
        similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    return KS @ np.linalg.pinv(A) @ KS.T


def ratio_nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error"):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    eps = 1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample( \
        list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig * np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)
    elif min_eig_mode == False:
        similarity_matrix_x = deepcopy(similarity_matrix)
    else:
        local_min_eig = min(0, np.min(np.linalg.eigvals(A))) - eps
        ratio = min_eig / local_min_eig
        A = (1.0 / ratio) * A - np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)

    KS = similarity_matrix_x[:, sample_indices]
    return KS @ np.linalg.pinv(A) @ KS.T


def nystrom_with_eig_estimate(similarity_matrix, k, return_type="error",
                              scaling=False, mult=0, eig_mult=1, new_rescale=True):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps = 1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample( \
        list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k * len(similarity_matrix)))
    else:
        large_k = min(mult * k, len(similarity_matrix) - 1)
    larger_sample_indices = np.sort(random.sample( \
        list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]
    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    min_eig = eig_mult * min_eig
    if scaling == True:
        min_eig = min_eig * np.float(len(similarity_matrix)) / np.float(large_k)

    A = A - min_eig * np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]

    if new_rescale:
        alpha = np.linalg.norm(A) / (np.linalg.norm(A - min_eig * np.eye(len(A))))
    else:
        alpha = 1.0

    return KS @ np.linalg.pinv(A) / alpha @ KS.T


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
    n, d = similarity_matrix.shape
    # setting up c, r, eps, and delta for error bound
    # c = (64*k*((1+8*np.log(1/delta))**2) / (eps**4)) + 1
    # r = (4*k / ((delta*eps)**2)) + 1
    # c = 4*k
    c = k
    r = k
    if c > n:
        c = n
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
        assert 1 <= k and k <= min(c, r)
    except AssertionError as error:
        print("1 <= k <= min(c,r)")

    # using uniform probability instead of row norms
    pj = np.ones(d).astype(float) / float(d)
    qi = np.ones(n).astype(float) / float(n)

    # choose samples
    samples_c = np.random.choice(range(d), c, replace=False, p=pj)
    if same:
        samples_r = samples_c
    else:
        samples_r = np.random.choice(range(n), r, replace=False, p=qi)

    # grab rows and columns and scale with respective probability
    samp_pj = pj[samples_c]
    samp_qi = qi[samples_r]
    C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj * c)
    rank_k_C = C
    # modification works only because we assume similarity matrix is symmetric
    R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi * r)
    R = R.T
    psi = C[samples_r, :].T / np.sqrt(samp_qi * r)
    psi = psi.T

    U = np.linalg.pinv(rank_k_C.T @ rank_k_C)
    # i chose not to compute rank k reduction of U
    U = U @ psi.T
    return (C @ U) @ R


def nystrom_with_eig_estimate_sub(similarity_matrix, k, return_type="error",
                                  scaling=False, mult=2, eig_mult=1.5, new_rescale=True):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps = 1e-16
    list_of_available_indices = range(len(similarity_matrix))
    # sample_indices = np.sort(random.sample(\
    #                  list_of_available_indices, k))
    # A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k * len(similarity_matrix)))
    else:
        large_k = min(mult * k, len(similarity_matrix) - 1)
    larger_sample_indices = np.sort(random.sample(
        list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]

    sample_indices = np.sort(random.sample(
        list(larger_sample_indices), k))
    A = similarity_matrix[sample_indices][:, sample_indices]

    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    min_eig = eig_mult * min_eig
    if scaling == True:
        min_eig = min_eig * np.float(len(similarity_matrix)) / np.float(large_k)

    A = A - min_eig * np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]

    if new_rescale:
        alpha = np.linalg.norm(A) / (np.linalg.norm(A - min_eig * np.eye(len(A))))
    else:
        alpha = 1.0

    return KS @ np.linalg.pinv(A)/alpha @ KS.T


# def CUR_alt(similarity_matrix, k, return_type="error"):
#     """
#     compute CUR approximation
#     versions:
#     U = S2^T K S1
#     """
#     list_of_available_indices = range(len(similarity_matrix))
#     sample_indices_rows = np.sort(random.sample(list_of_available_indices, min(k, len(similarity_matrix))))
#     sample_indices_cols = np.sort(random.sample(list(sample_indices_rows), int(k / 2)))
#     A = similarity_matrix[sample_indices_rows][:, sample_indices_cols]
#
#     similarity_matrix_x = deepcopy(similarity_matrix)
#     KS_cols = similarity_matrix_x[:, sample_indices_cols]
#     KS_rows = similarity_matrix_x[sample_indices_rows]
#     return  KS_cols @ np.linalg.pinv(A) @ KS_rows


def CUR_alt(similarity_matrix, k, mult=2, return_type="error"):
    """
    compute CUR approximation
    versions:
    U = S2^T K S1
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices_rows = np.sort(random.sample(list_of_available_indices, min(k, len(similarity_matrix))))
    if mult > 1:
        sample_indices_cols = np.sort(random.sample(list(sample_indices_rows), int(k / mult)))
    else:
        sample_indices_cols = np.sort(random.sample(list_of_available_indices, min(k, len(similarity_matrix))))
    A = similarity_matrix[sample_indices_rows][:, sample_indices_cols]

    similarity_matrix_x = deepcopy(similarity_matrix)
    KS_cols = similarity_matrix_x[:, sample_indices_cols]
    KS_rows = similarity_matrix_x[sample_indices_rows]
    return  KS_cols @ np.linalg.pinv(A) @ KS_rows