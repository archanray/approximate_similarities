import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys
from copy import copy
from Nystrom import simple_nystrom
from copy import deepcopy
from utils import read_file, read_labels
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random

def CUR_alt(similarity_matrix, k, return_type="error"):
    """
    compute CUR approximation
    versions:
    U = S2^T K S1
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices_rows = np.sort(random.sample(\
                     list_of_available_indices, min(k, len(similarity_matrix))))
    sample_indices_cols = np.sort(random.sample(\
                     list(sample_indices_rows), int(k/2)))
    A = similarity_matrix[sample_indices_rows][:, sample_indices_cols]
    
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS_cols = similarity_matrix_x[:, sample_indices_cols]
    KS_rows = similarity_matrix_x[sample_indices_rows]
    if return_type == "approximation":
        return (KS_cols @ np.linalg.pinv(A)) @ KS_rows
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS_cols @ np.linalg.pinv(A) @ KS_rows)\
                / np.linalg.norm(similarity_matrix)

def CUR(similarity_matrix, k, eps=1e-3, delta=1e-14, return_type="error", same=True):
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
    if return_type == "approximation":
        return (C @ U) @ R
    if return_type == "error":
        # print(np.linalg.norm((C @ U) @ R))
        relative_error = np.linalg.norm(similarity_matrix - \
            ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
        return relative_error

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
    if return_type == "approximation":
        return KS @ np.linalg.pinv(A) @ KS.T


def nystrom_with_eig_estimate_sub(similarity_matrix, k, return_type="error", \
    scaling=False, mult=0, eig_mult=1):
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

    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "approximation":
        return KS @ np.linalg.pinv(A) @ KS.T

    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def error_compute(x1, x2):
    return np.linalg.norm(x1-x2)

def add_rows_back(predictions, duplicate_ids, matched_ids):
    # first add rows and columns
    complete_predictions = copy(predictions)
    for i in range(len(duplicate_ids)):
        # print("duplicate id:", duplicate_ids[i])
        current_id = duplicate_ids[i]
        complete_predictions = np.insert(complete_predictions, current_id, np.zeros(complete_predictions.shape[1]), 0)
        complete_predictions = np.insert(complete_predictions, current_id, np.zeros(complete_predictions.shape[0]), 1)
    # print(complete_predictions.shape)
    for i in range(len(duplicate_ids)):
        current_id = duplicate_ids[i]
        matching_id = matched_ids[i]
        complete_predictions[current_id, :] = complete_predictions[matching_id, :]
        complete_predictions[:, current_id] = complete_predictions[:, matching_id]
    return complete_predictions

def find_matches(reshaped_preds, indices):
    # needs to be filled in
    all_indices = list(range(len(reshaped_preds)))
    missing_indices = list(set(all_indices) - set(indices))
    missing_indices = np.sort(missing_indices)
    # first find the exact copies 
    exact_copies = []
    for i in range(len(missing_indices)):
        to_find = reshaped_preds[missing_indices[i],:]
        match_ids = (reshaped_preds == to_find).all(axis=1).nonzero()
        exact_copies.append(match_ids[0])
    # remove the IDs which occur in missing indices for each copies 
    # (this will leave out the ones left in reshaped_preds)
    for i in range(len(exact_copies)):
        removed_duplicates = list(set(exact_copies[i]) - set(missing_indices))[0]
        exact_copies[i] = removed_duplicates
    exact_copies = np.array(exact_copies)
    return missing_indices, exact_copies

def compare_results(A, original_score, method="pearson"):
    true_value = original_score[:,2]
    rows = original_score[:,0].astype(int)
    cols = original_score[:,1].astype(int)
    computed_value = A[rows, cols]

    if method == "pearson":
        results = pearsonr(true_value, computed_value)[0]
    if method == "spearman":
        results = spearmanr(true_value, computed_value)[0]
    if method == "f1":
        computed_value = np.floor(computed_value)
        results = f1_score(true_value, computed_value)
    if method == "accuracy":
        computed_value = 1 - computed_value
        computed_value = np.floor(computed_value)
        # print(true_value, computed_value)
        results = accuracy_score(true_value, computed_value)
    return results

def scale_scores(score1, score2, original_score):
    rows = original_score[:,0].astype(int)
    cols = original_score[:,1].astype(int)
    computed_value = score2[rows, cols]
    min_val_o = np.min(computed_value)
    max_val_o = np.max(computed_value)
    
    min_val_n = np.min(score1[rows, cols])
    max_val_n = np.max(score1[rows, cols])
    # normalize between 0 and 1
    score1[rows, cols] = (score1[rows, cols] - min_val_n) / max_val_n
    score1[rows, cols] = max_val_o*score1[rows, cols] + min_val_o
    return score1

def evaluate(A, original_score, database="stsb"):
    if database == "stsb":
        print("pearson score:", \
            compare_results(A, original_score, method="pearson"))
        print("spearman score:", \
            compare_results(A, original_score, method="spearman"))
        pass
    if database == "mrpc":
        print("F1 score:", \
            compare_results(A, original_score, method="f1"))
        pass
    if database == "rte":
        print("accuracy score:", \
            compare_results(restored_similarity_matrix, original_scores, method="accuracy"))
        pass
    return None

# def fix_vals_first(A, original_score):
#     rows = original_score[:,0].astype(int)
#     cols = original_score[:,1].astype(int)
#     computed_scores = A[rows, cols]
#     D = np.eye(len(A))
#     D[rows, cols] = computed_scores
#     D[cols, rows] = computed_scores
#     return D

#############################################################################################
#================================= Data extraction =========================================#
# READ THE FILE
dataset = sys.argv[1]
num_samples = int(sys.argv[3])

reshaped_preds = read_file(file_="../GYPSUM/"+dataset+"_predicts_0.npy")
# READ LABELS FROM FILE
original_scores = read_labels("../GYPSUM/"+dataset+"_label_ids.txt")

# reshaped_preds = reshaped_preds
# reshaped_preds = fix_vals_first(reshaped_preds, original_scores)
# print(reshaped_preds.shape)

# FIND OUT DUPLICATE ROWS AND ELIMINATE!!
unique_rows, unique_indices = np.unique(reshaped_preds, axis=0, return_index=True)

# FIND DUPLICATE ROWS AND ID THEM WITH MATCHING ROWS
# Duplicate ids == rows in reshaped_preds which has been removed
# Matched ids = rows in reshaped_preds which has a duplicate in duplcate_ids
unique_indices = np.sort(unique_indices)
# checked, and the following line works fine
duplicate_ids, matched_ids = find_matches(reshaped_preds, unique_indices)


# Reshape the predictions to remove duplicate rows
similarity_matrix = copy(reshaped_preds[unique_indices][:, unique_indices])
similarity_matrix = (similarity_matrix+similarity_matrix.T) / 2.0
# restore symmetrized matrix to original shape
restored_similarity_matrix = add_rows_back(similarity_matrix, duplicate_ids, matched_ids)
#############################################################################################

#============================ check the required scores ================================#
# print("true scores")
# evaluate(reshaped_preds, original_scores, database=dataset)
# print("symmetrized scores")
# evaluate(restored_similarity_matrix, original_scores, database=dataset)
# print(np.min(restored_similarity_matrix), np.max(restored_similarity_matrix), \
#     np.min(original_scores[:,2]), np.max(original_scores[:,2]))
# print("original error")
# print(np.linalg.norm(reshaped_preds-restored_similarity_matrix)/np.linalg.norm(reshaped_preds))
#########################################################################################

#================================== approximation =====================================#
mode = sys.argv[2]
if mode == "nystrom":
    approx_sim_mat = nystrom_with_eig_estimate(\
                    similarity_matrix, num_samples, return_type="approximation", \
                    mult=2, eig_mult=1.5)
if mode == "CUR":
    approx_sim_mat = CUR(similarity_matrix, num_samples, return_type="approximation")

if mode == "CUR_alt":
    approx_sim_mat = CUR_alt(similarity_matrix, num_samples, return_type="approximation")

approx_sim_mat = add_rows_back(approx_sim_mat, duplicate_ids, matched_ids)

print("error:", 
    np.linalg.norm(reshaped_preds-approx_sim_mat)/ np.linalg.norm(reshaped_preds))
#########################################################################################

#============================ compute approximation error ==============================#
# print("approximate similarity scores "+str(num_samples))
# print(np.min(approx_sim_mat), np.max(approx_sim_mat), \
#     np.min(original_scores[:,2]), np.max(original_scores[:,2]))
evaluate(approx_sim_mat, original_scores, database=dataset)
# evaluate(reshaped_preds, original_scores, database=dataset)
#########################################################################################

#================================= checking for RTE ====================================#
# grab the scores
# rows = original_scores[:,0].astype(int)
# cols = original_scores[:,1].astype(int)
# computed_scores_forward = reshaped_preds[rows, cols]
# computed_scores_backward = reshaped_preds[cols, rows]
# print(pearsonr(computed_scores_forward, computed_scores_backward)[0])
# print(spearmanr(computed_scores_forward, computed_scores_backward)[0])
#########################################################################################