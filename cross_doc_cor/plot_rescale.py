

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_errors(name="ECB+", \
    save_path="plot", y_lims=[]):
    # res_f1 = {
    #     "exact": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [38.74880657037666] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "sym": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [38.623556216790654]* len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "nystrom": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [29.453001115656377, 31.901301909123653, 33.89523242300988, 33.60986202955343, 35.574965261695226,
    #          37.107572264585926, 36.68823043790381, 36.770537450998, 37.259855174849875]
    #     ],
    #     "cur": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [24.150488063329803, 29.53522903033908, 32.426787836304165, 32.86921232537402, 34.89727972947586,
    #          34.90617098275133, 35.335144224069865, 36.590698518385224, 37.67883254117065]
    #     ],
    #     "cur_alt": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [28.128409441870726, 32.30474413561053, 34.43464486443086, 35.899560666636724, 37.12608083184479,
    #          36.29640492099795, 37.8827303695399, 38.26506415471348, 38.20478221078221]
    #     ],
    #     "nystrom_eig_est": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [20.222374744661096, 24.979276825939138, 28.922908462203445, 29.71230751996415, 29.92687277622848,
    #          29.973995518316446, 30.454572490706322, 30.282750331058367, 29.97471384518894]
    #     ],
    # }
    # res_err = {
    #     "exact": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.0] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "sym": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.12959994703820654] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "nystrom": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [2.772682744199066, 5.949554995156294, 2.930434588330667, 2.1836144439835365, 2.107271635851369,
    #          0.9578672924985379, 1.1370387264340296, 32.76028259180046, 0.5801060841147075]
    #     ],
    #     "cur": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [1.213165380442959, 1.053327605619403, 0.9190546640682088, 0.8326280683943349, 0.7526685722469477,
    #          0.6670065633334793, 0.5831746612592008, 0.473997983636062, 0.35587686046244565]
    #     ],
    #     "cur_alt": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.5711757211744267, 0.41354485861670676, 0.3593225652064925, 0.2901445826254471, 0.2558885392384951,
    #          0.23427848603048979, 0.20278928727650455, 0.1839981588473012, 0.16631205284245257]
    #     ],
    #     "nystrom_eig_est": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.5689465823771407, 0.5047999530139534, 0.4730218680756465, 0.4906711952185969, 0.5398778758939089,
    #          0.5763011570974493, 0.8108787254776367, 1.700433880149526, 6.527033157149613]
    #     ],
    # }

    # res_f1 = {
    #     "exact": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [38.74880657037666]* len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "sym": [
    #                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [38.623556216790654] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #     ],
    #     "nystrom": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [29.273947579996786, 32.55142694636823, 33.55880074524416, 33.35289997031761, 35.1294254709447,
    #          35.545589500727395, 35.89841655172414, 37.77288070512261, 37.72731167793039]
    #     ],
    #     "cur": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [26.017948958822608, 28.85030708792121, 32.50300643166974, 33.573281386967956, 33.91302603664417,
    #          34.19815229170651, 35.23213310341613, 35.88255328881163, 37.03267704164814]
    #     ],
    #     "cur_alt": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [25.62431259685632, 32.69287929549153, 34.440562468199275, 35.33895131428571, 37.223023204999336,
    #          36.73706224213655, 37.44623570608662, 37.7505733663037, 38.32921838883657]
    #     ],
    #     "nystrom_eig_est": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [21.15712345613003, 23.730934887256144, 21.220076372828483, 22.23987812632471, 23.44314492900952,
    #          25.636659374896425, 26.913771585557303, 28.20003487696183, 29.220036877689]
    #     ],
    # }
    # res_err = {
    #     "exact": [
    #         [1.0],
    #         [0.0]
    #     ],
    #     "sym": [
    #         [1.0],
    #         [0.12959994703820654]
    #     ],
    #     "nystrom": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [1.2391063137144813, 2.6258636582489165, 27.518758450522196, 372.7915584593121, 2.471352822742714,
    #          12.022232185339764, 476.2765504906115, 2.512993007886926, 5.5195300392917215]
    #     ],
    #     "cur": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [1.1410597881763231, 1.0401030437425554, 0.9344839214858597, 0.8482359069393325, 0.75524149975784,
    #          0.6741148013832006, 0.5747109803557344, 0.4611982041891614, 0.32543252483109225]
    #     ],
    #     "cur_alt": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.5807466922346248, 0.42206726787464655, 0.35917188020374813, 0.2994045708100829, 0.26552745878988604,
    #          0.2386693394382009, 0.201705416933918, 0.18038133903443856, 0.17104379934745992]
    #     ],
    #     "nystrom_eig_est": [
    #         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #         [0.569596063237659, 0.5113661454943242, 0.526472382885246, 0.5107179394033644, 0.4976326959651174,
    #          0.47559882585276697, 0.45627401598865824, 0.44993160530046883, 0.4465001946793235]
    #     ],
    # }

    res_f1 = {
        "exact": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [38.74880657037666] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ],
        "sym": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [38.623556216790654] * len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ],
        "nystrom": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [28.66996569658072, 31.069456184840803, 33.96952487441257, 33.16178318539943, 35.049715066170585,
             36.03546386711288, 36.22893609262467, 36.86991166525152, 37.979235256743486]
        ],
        "cur": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [26.0632491186839, 30.69291804979253, 31.1141873924605, 32.64047932018854, 33.27424428405428,
             35.114662422568614, 36.4912790034519, 36.05007852965748, 37.8609048136084]
        ],
        "cur_alt": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [26.314911274615888, 30.58182837080548, 34.13020143616526, 36.21210791324038, 37.25762167325142,
             36.96296915562987, 36.88764789356985, 38.907106769956215, 38.290053915276]
        ],
        # "nystrom_eig_est": [
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     [21.979051420767895, 25.352385200202736, 27.06278413311544, 29.196866279906022, 29.018682926829268,
        #      30.612333058532563, 29.179954679595284, 30.200186931769906, 30.887856446461587]
        # ],
        "sms_nystrom": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [23.049305901497792, 23.354157508229925, 23.189379573327837, 22.967590968076824, 24.077846873289893,
             25.54213553511981, 26.87054748275356, 28.10449818117719, 29.08939030628557]
        ],
        # "nystrom_eig_est_rescale": [
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     [28.083729411172115, 32.27856120599179, 33.38746050919904, 35.4499330979095, 35.14915692140332,
        #      36.257137139807895, 35.49082717063818, 35.72159248585761, 34.8295015856487]
        # ],
        "sms_nystrom_rescale": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [28.932765556151733, 31.245601914863656, 32.708575561036504, 32.83120725325544, 32.34529857022709,
             34.62388378245079, 35.231566758706684, 36.014258663965826, 36.22113445130921]
        ],
    }

    res_err = {
        "exact": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.0]* len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ],
        "sym": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.12959989021841523]* len([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ],
        "nystrom": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [4.342220716667822, 11.275444166490063, 8.263790517132321, 5.532736135583545, 3.0860380568041528,
             6.330362742700866, 3.0820017163753017, 0.7559040721936414, 0.4881327266381568]
        ],
        "cur": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [1.2506850299651378, 0.9821308262273527, 0.9499660660723231, 0.8348161438286801, 0.7669418908063592,
             0.6627484459433014, 0.5623646705331076, 0.45554272193861695, 0.3117326505758718]
        ],
        "cur_alt": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.6004475882086973, 0.43179247245457153, 0.32093809253024, 0.2726549842176892, 0.2382369922122965,
             0.1934738589564467, 0.1643830599440037, 0.13495955721216663, 0.10428315017308167]
        ],
        # "nystrom_eig_est": [
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     [0.5618655930746014, 0.518958921144466, 0.45961138973690946, 0.5124047318833038, 0.5155344583602824,
        #      0.8141606750568168, 5.623518004600581, 2.3492756721037114, 1.9044613243090371]
        # ],
        "sms_nystrom": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.5553241733711308, 0.511197591938049, 0.497074373296004, 0.48909292310611563, 0.4855289484960187,
             0.46116892172897556, 0.44068697032387516, 0.4346457718390586, 0.4354340854611954]
        ],
        # "nystrom_eig_est_rescale": [
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #     [0.5008056708905144, 1.2540112639010397, 0.5110824449576313, 0.5723155207452285, 0.6368990025335101,
        #      3.3149141171732714, 2.7882566305802, 3.879277388059546, 10.267227824099862]
        # ],
        "sms_nystrom_rescale": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.4726508734172051, 0.42234218989358896, 0.439497354165857, 0.41485995975506573, 0.4232629464834134,
             0.4573009746496563, 0.4814687377786, 0.5258590908690741, 0.5798125092230011]
        ],
    }

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
    plt.rc('legend', fontsize=18)

    STYLE_MAP = {"exact": {"color": "#505050", 'linestyle': '--', 'label': 'Exact', 'linewidth': 1},
                 "sym": {"color":  "#BEBEBE", 'linestyle': '-.', 'label': 'Symm.', 'linewidth': 1},
                 "cur": {"color": '#9A0EEA', "marker": "v", "markersize": 7, 'label': 'SiCUR', 'linewidth': 1},
                 "cur_alt": {"color": "#DAA520", "marker": "o", "markersize": 7, 'label': 'StaCUR', 'linewidth': 1},
                 "nystrom": {"color": "#FC5A50",  "marker": "^", "markersize": 7, 'label': 'Nystrom (No Rescale)', 'linewidth': 1},
                 "nystrom_eig_est": {"color": "#069AF3", "marker": ".", "markersize": 7, 'label': 'min-eig Nystrom (Rescale)', 'linewidth': 1},
                 "sms_nystrom": {"color": "#509cfc", "marker": "s", "markersize": 7, 'label': 'SMS-Nystrom (No Rescale)',
                                     'linewidth': 1},
                 "nystrom_eig_est_rescale": {"color": "#069AF3", "marker": ".", "markersize": 7, 'label': 'min-eig Nystrom  (Rescale)',
                                     'linewidth': 1},
                 "sms_nystrom_rescale": {"color": "#15B01A", "marker": "s", "markersize": 7, 'label': 'SMS-Nystrom',
                                 'linewidth': 1},

                 }

    for m,arr in res_f1.items():
        ax1.plot(np.array(arr[0]), arr[1], **STYLE_MAP[m], alpha=0.5)

    title_name = 'ECB+'
    directory = "figures/"+save_path+"/"
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename=name.lower() + '_f1'
    path = os.path.join(directory, filename+".pdf")

    plt.locator_params(axis='x', nbins=6)
    if len(y_lims) > 0:
        plt.ylim(bottom=y_lims[0], top=y_lims[1])
    plt.xlabel("Proportion of dataset chosen as landmark samples", fontsize=15)
    plt.ylabel("CoNLL F1", fontsize=15)
    plt.title(title_name, fontsize=21)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # ax1.legend(loc='lower right')
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax1.legend(loc='lower right', #bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=False, shadow=False,prop={'size': 8})
    plt.savefig(path)
    plt.gcf().clear()

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
    plt.rc('legend', fontsize=18)

    for m, arr in res_err.items():
        ax1.plot(np.array(arr[0]), arr[1], **STYLE_MAP[m], alpha=0.5)

    title_name = 'ECB+'
    directory = "figures/" + save_path + "/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename = name.lower() + '_err'
    path = os.path.join(directory, filename + ".pdf")

    plt.locator_params(axis='x', nbins=6)
    plt.ylim(bottom=0, top=2)
    plt.xlabel("Proportion of dataset chosen as landmark samples", fontsize=15)
    plt.ylabel("Approx. Error", fontsize=15)
    plt.title(title_name, fontsize=21)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # ax1.legend(loc='upper left')
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.legend(loc='upper left',  # bbox_to_anchor=(0.5, 1.05),
               ncol=3, fancybox=False, shadow=False, prop={'size': 8})
    plt.savefig(path)
    plt.gcf().clear()

plot_errors()