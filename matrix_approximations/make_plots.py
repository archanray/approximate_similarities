from plotter import plot_errors
import os
import sys
from glob import glob
import pickle

# files = glob("approx_exps/"+"*.pkl")


# for file_ in files:
#     print(file_)

#     with open(file_, "rb") as f:
#         # lists = pickle.load(f)
#         # print(len(lists))
#         true_error, nscaling_error_list, CUR_same_error_list, CUR_diff_error_list, \
#             SiCUR_error_list, SkCUR_error_list = pickle.load(f)
#     f.close()

#     file_ = file_.split("\\")[-1]
#     nom = file_.split("_")[0]
#     # nom = file_.split("_")[0]

#     if nom == "20ng2":
#         nom = "news"

#     if nom == "rte" or nom == "mrpc":
#         id_count = len(true_error)*10 + 10
#     else:
#         id_count = 1500
#     min_y = 0.0
#     if nom == "news":
#         min_y = 0.2
#     max_y = 0.5

#     if nom == "news" or nom == "mrpc" or nom == "recipe":
#         max_y = 0.5

#     plot_errors([true_error, nscaling_error_list, CUR_same_error_list, CUR_diff_error_list, \
#             SiCUR_error_list, SkCUR_error_list], id_count, \
#                  ["Nystrom", "SMS-Nystrom", "StaCUR(s)", "StaCUR(d)", "StaCUR", "SkeletonApprox"], \
#                  step=10, \
#                  colormaps=1, name=nom, \
#                  save_path="comparison_all_versions", \
#                  y_lims=[min_y, max_y])


files = glob("post_exp/"+"*.pkl")
def name_corrector_and_idcount(file_, true_nystrom):
    file_ = file_.split("\\")[-1]
    name = file_.split("_")[0]
    if name == "20ng2":
        name = "news"
    if name == "rte" or name == "mrpc" or name=="PSD":
        id_count = len(true_nystrom)*10 + 10
    else:
        id_count = 1500
    return name, id_count

mode = "nystrom vs cur variants"
for file_ in files:
    print(file_)

    with open(file_, "rb") as f:
        data = pickle.load(f)
    f.close()

    if mode == "nystrom v cur":
        true_nystrom = data[0]
        CUR = data[1]

        name, id_count = name_corrector_and_idcount(file_, true_nystrom)
        print(name, len(CUR))
        plot_errors([true_nystrom, CUR], \
                     id_count, \
                     ["true nystrom", "CUR"], \
                     step=10, \
                     colormaps=1, name=name, \
                     save_path="comparison_true_nystrom_vs_CUR", y_lims=[0.0, 1.0])

    if mode == "nystrom vs cur variants":
        true_error = data[0]
        sms = data[1]
        CURs = data[2]
        CURd = data[3]
        SiCUR = data[4]
        SkCUR = data[5]

        name, id_count = name_corrector_and_idcount(file_, true_error)

        if name == "twitter" or name == "PSD" or name == "ohsumed":
            y_lim = [0.0, 0.8]
        else:
            y_lim = [0.0, 2.0]


        plot_errors([true_error, sms, CURs, CURd, SiCUR, SkCUR], \
                     id_count, \
                     ["Nystrom", "SMS-Nystrom", "StaCUR(s)", "StaCUR(d)", "SiCUR", "SkelApprox"], \
                     step=10, \
                     colormaps=1, name=name, \
                     save_path="post_exp_plots", y_lims=y_lim)

    if mode == "nystom only":
        true_nystrom = data[0]

        name, id_count = name_corrector_and_idcount(file_, true_nystrom)

        if name == "PSD" or name == "twitter":
            y_lims=[]
        else: 
            y_lims=[0.0,2.0]

        plot_errors([true_nystrom], \
                     id_count, \
                     ["true nystrom"], \
                     step=10, \
                     colormaps=1, name=name, \
                     save_path="true_nystrom", y_lims=y_lims)