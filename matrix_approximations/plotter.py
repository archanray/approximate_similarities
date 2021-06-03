import matplotlib.pyplot as plt
import numpy as np
import os

def plot_errors(lists, id_count, labels, step=10, colormaps=1, name="MRPC", \
    save_path="comparison_among_eigenvalues_and_z", y_lims=[]):


    x_axis = list(range(10, id_count, step))
    print(name)
    if name == "stsb":
        total_samples = 3000.0
    if name == "mrpc":
        total_samples = 816.0
    if name == "rte":
        total_samples = 554.0
    if name == "twitter":
        total_samples = 2176.0
    if name == "recipe":
        total_samples = 27841.0
    if name == "ohsumed":
        total_samples = 3999.0
    if name == "news":
        total_samples = 11293.0
    if name == "PSD":
        total_samples = 1000.0
    x_axis = np.array(x_axis) / total_samples

    plt.gcf().clear()
    scale_ = 0.55
    new_size = (scale_ * 10, scale_ * 8.5)
    plt.gcf().set_size_inches(new_size)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=13)

    STYLE_MAP = {"plot0":{"marker":"^", "markersize":7, "linewidth":1},\
                 "plot1":{"marker":"v", "markersize":7, "linewidth":1},
                 "plot2":{"marker":"o", "markersize":7, "linewidth":1},
                 "plot3":{"marker":"*", "markersize":7, "linewidth":1},
                 "plot4":{"marker":".", "markersize":7, "linewidth":1}}

    for i in range(len(lists)):
        error_pairs = lists[i]
        arr1 = np.array(error_pairs)
        ax1.plot(np.array(x_axis),arr1,\
            label=labels[i], **STYLE_MAP["plot4"], alpha=0.5)

    # if colormaps == 1:
    #     colormap = plt.cm.cool
    #     colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    colors = ["#FC5A50", "#069AF3", "#15B01A", "#9A0EEA", "#DAA520", "#580F41"]

    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    title_name = name
    if title_name == "stsb":
        title_name = "STS-B"
    if title_name == "mrpc":
        title_name = "MRPC"
    if title_name == "rte":
        title_name = "RTE"
    if title_name == "twitter":
        title_name = "Twitter"
    if title_name == "recipe":
        title_name = "RecipeL"
    if title_name == "news":
        title_name = "20-News"
    if title_name == "ohsumed":
        title_name = "Ohsumed"

    directory = "figures/"+save_path+"/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename=name.lower()
    path = os.path.join(directory, filename+".pdf")


    plt.locator_params(axis='x', nbins=6)
    if len(y_lims) > 0:
        plt.ylim(bottom=y_lims[0], top=y_lims[1])
    plt.xlabel("Proportion of dataset chosen as landmark samples", fontsize=15)
    plt.ylabel("Average approximation error", fontsize=15)
    plt.title(title_name, fontsize=21)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax1.legend(loc='upper right')
    plt.savefig(path)
    plt.gcf().clear()