import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys

def read_file(file_="predicts_0.npy"):
        """
        input: number o9f unique sentences
        output: N x N matrix of similarities among sentences
        file name is fixed for now, can be changed
        """
        file_to_read = file_
        predictions = np.load(file_to_read)
        if predictions.shape[1] == 2:
            predictions_ = np.argmax(predictions, axis=1)
        else:
            predictions_ = predictions
        pred_id_count = int(np.sqrt(predictions_.shape[0]))
        reshaped_preds = np.reshape(predictions_, (-1, pred_id_count))
        return reshaped_preds

def read_fileGPU(file_="predicts_0.npy"):
        """
        input: number o9f unique sentences
        output: N x N matrix of similarities among sentences
        file name is fixed for now, can be changed
        """
        import torch
        reshaped_preds = read_file(file_)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy(reshaped_preds).to(device)


def read_mat_file(file_="twitter_K_set1.mat", version="default", return_type="mat_only", mode="train"):
        """
        input: mat file
        output: NxN matrix of similarities among sentences
        file name can be changed
        """
        if version == "default":
            print("version: default")
            import scipy.io as sio
            mat = sio.loadmat(file_)
            if mode == "train":
                reshaped_preds_full = mat['trainData']
            if mode == "test":
                reshaped_preds_full = mat['testData']
            reshaped_preds = reshaped_preds_full[:, 1:]
            pass
        if version == "v7.3":
            print("version: H5PY")
            import h5py
            f = h5py.File(file_, 'r')
            if mode == "train":
                reshaped_preds_full = f.get("trainData")
            if mode == "test":
                reshaped_preds_full = f.get("testData")
            reshaped_preds_full = np.array(reshaped_preds_full)
            reshaped_preds_full = reshaped_preds_full.T
            reshaped_preds = reshaped_preds_full[:, 1:]
            pass
        if return_type == "all":
            return reshaped_preds, reshaped_preds_full[:, 0]
        return reshaped_preds


def read_mat_fileGPU(file_="twitter_K_set1.mat", version="default"):
        import torch
        reshaped_preds = read_mat_file(file_, version)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy(reshaped_preds).to(device)

def read_labels(file_="label_ids.txt"):
        """
        input: sentence ID pairs and corresponding matching score
        output: 3 lists:: two of sentence IDs and one for scores
        """
        file_to_read = file_
        f = open(file_to_read, "r")
        all_lines = f.readlines()
        f.close()
        all_lines = [x.strip("\n") for x in all_lines]
        source = []
        target = []
        match = []
        for i in range(len(all_lines)):
                s,t,m = all_lines[i].split()
                # print(s,t,m)
                source.append(int(s))
                target.append(int(t))
                match.append(float(m))

        # organize in a matrix
        original_scores = np.zeros((len(source), 3))
        original_scores[:,0] = source
        original_scores[:,1] = target
        original_scores[:,2] = match

        return original_scores

def is_pos_def(x):
        """
        checks if a matrix is PD
        """
        return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semi_def(x):
        """
        checks if a matrix is PSD
        """
        return np.all(np.linalg.eigvals(x) >= 0)

def is_real_eig(x):
        """
        checks if all eigenvalues of x are real
        """
        return np.all(np.isreal(x))

def row_norm_matrix(K):
        """
        row normalize a matrix
        """
        diag = np.diagonal(K)
        _len = len(diag)
        rank_l_K = K / diag.reshape((_len, 1))
        return rank_l_K

def laplacian_norm_matrix(K, tilde_K):
        """
        compute laplacian norm of a matrix
        """
        #print(np.diagonal(tilde_K))
        tilde_diag = np.diagonal(tilde_K) + 1e-5
        tilde_diag = np.sqrt(tilde_diag) 
        tilde_diag = 1./tilde_diag
        diag = np.diagonal(K)
        diag = np.sqrt(diag)
        
        tilde_D = np.zeros_like(K)
        np.fill_diagonal(tilde_D, tilde_diag)
        D = np.zeros_like(K)
        np.fill_diagonal(D, diag)

        K_bar = D @ tilde_D @ tilde_K @ tilde_D @ D
        return K_bar


def norm_diag(K):
        """
        normalize diagonal of a matrix
        """
        x = np.diagonal(K)
        # x = x + 1e-10 # accounting for numerical instability
        x = np.sqrt(x)
        x = x + 1e-10
        x = 1./x # accounting for numerical instability
        D = np.zeros_like(K)
        np.fill_diagonal(D, x)
        K = (D.T @ K) @ D
        return K

def viz_diagonal(A, mat_type="original"):
        """
        visualize the diagonal of a matrix
        """
        diag = np.diagonal(A)
        sns.set()
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        cmap = ListedColormap(sns.color_palette(flatui).as_hex())
        x = list(range(len(diag)))
        fig = plt.gcf()
        fig.set_size_inches(20,12, forward=True)
        plt.plot(x, diag, label="diagonal")
        plt.xlabel("number of samples")
        plt.ylabel("diagonal values")
        plt.legend(loc="upper right")
        plt.title("plot of diagonal values")
        plt.savefig("figures/diagonal_"+mat_type+".pdf")
        plt.clf()
        return None 

def viz_eigenvalues(A, name="test"):
        """
        visualize eigenvalues of a matrix
        """
        eigenvals = np.linalg.eigvals(A)

        abs_eigvals = np.absolute(eigenvals)
        real_eigvals = np.real(eigenvals)

        # create the plot
        sns.set()
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        cmap = ListedColormap(sns.color_palette(flatui).as_hex())
        x = list(range(len(abs_eigvals)))
        fig = plt.gcf()
        fig.set_size_inches(20, 12, forward=True)
        plt.scatter(x, abs_eigvals, label="absolute values", c='#e74c3c')
        plt.scatter(x, real_eigvals, label="real part", c='#34495e')
        plt.figtext(.7, .7, "PSD? = "+str(is_pos_def(A)))
        plt.figtext(.7, .65, "Rank = "+str(np.linalg.matrix_rank(A)))
        plt.xlabel("eigenvalue indices")
        plt.ylabel("values")
        plt.legend(loc='upper right')
        plt.title("plot of the distribution of norm and real part of the eigenvalues")
        plt.savefig("figures/eigenvalue"+name+".pdf")
        plt.clf()
        return None

def lev_plot(data, dataset):
    x_axis = list(range(len(data)))

    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=11)

    STYLE_MAP = {"data": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'scores', 'linewidth': 1},
            }

    plt.gcf().clear()
    scale_ = 0.55
    new_size = (scale_ * 10, scale_ * 8.5)
    plt.gcf().set_size_inches(new_size)

    title_name = dataset

    data_pairs = [(x, y) for x, y in zip(x_axis, data)]
    arr1 = np.array(data_pairs)
    plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['data'])
    plt.locator_params(axis='x', nbins=6)
    plt.xlabel("Samples")
    plt.ylabel("Leverage scores")
    plt.title(title_name, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend(loc='upper right')
    plt.savefig("figures/leverage_score_plot.pdf")
    # plt.savefig("./test1.pdf")
    plt.gcf().clear()
    
    return None
