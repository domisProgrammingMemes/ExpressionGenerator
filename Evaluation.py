from sys import path
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

AUs = ["AU1L", "AU1R", "AU2L", "AU2R", "AU4L", "AU4R", "AU6L", "AU6R", "AU9", "AU10", "AU13L", "AU13R", "AU18",
       "AU22", "AU27"]
AUs_reduced = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU10", "AU13", "AU18", "AU22", "AU27"]
colors = ["#ff4a47", "#ff4a47", "#fa8b2a", "#fa8b2a", "#d1ce00", "#d1ce00", "#279c00", "#279c00", "#00eeff", "gray",
          "#a600ff", "#a600ff", "black", "#007bff", "red"]
colors_reduced = ["#ff4a47", "#fa8b2a", "#d1ce00", "#6aff00", "#00eeff", "gray", "#a600ff", "black", "#007bff", "red"]
y_indexes = np.arange(len(AUs))
y_indexes_reduced = np.arange(len(AUs_reduced))


def compare_two_sequences(GT: str, ExGen: str):

    data = {}
    data2 = {}
    GT_data = pd.read_csv(GT)
    ExGen_data = pd.read_csv(ExGen)
    duration = GT_data.iloc[-1, 0]

    # ====
    # Data
    # ====
    data[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data[i] = GT_data.iloc[:duration, i]

    array = [data[i].values for i in range(0, 16)]

    data2[0] = data[0]
    # data2[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data2[i] = ExGen_data.iloc[:duration, i]

    array2 = [data2[i].values for i in range(0, 16)]


    # gs = GridSpec(1, 2, figure=fig)

    # ===========================
    # first subplot: Ground Truth
    # ===========================
    # ax = fig.add_subplot(gs[0, 0], projection="3d")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1, zdir="y", color=colors[i - 1])

    # ax.set_title("Im Datensatz", y=.84, pad=0, fontsize=15)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)

    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # ====================================
    # second subplot: Expression Generator
    # ====================================
    # ax = fig.add_subplot(gs[0, 1], projection="3d")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, 16):
        ax.plot(array2[0], array2[i], i - 1, zdir="y", color=colors[i - 1])

    # ax.set_title("Variation", y=.84, pad=0, fontsize=15)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)


    # fig.subplots_adjust(top=0.985,
    #                     bottom=0.015,
    #                     left=0.008,
    #                     right=0.991,
    #                     hspace=0.2,
    #                     wspace=0.018)
    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # ====================================
    # third plot: 2 in 1
    # ====================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    width = 0.0
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1 + width, zdir="y", color=colors[i - 1], linestyle="--")
        ax.plot(array2[0], array2[i], i - 1, zdir="y", color=colors[i - 1])
        ax.add_collection3d(plt.fill_between(array[0], array[i], array2[i], color=colors[i-1], alpha=0.1), zs=i-1, zdir='y')

    # def polygon_under_graph(xlist, ylist):
    #     """
    #     Construct the vertex list which defines the polygon filling the space under
    #     the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    #     """
    #     return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]
    #
    # # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
    # verts = []
    #
    # zs = range(0, 15)
    #
    # for i in zs:
    #     verts.append(polygon_under_graph(array[0], array2[i+1]))
    #
    # poly = PolyCollection(verts, facecolors=colors, alpha=.6)
    # ax.add_collection3d(poly, zs=zs, zdir='y')


    # ax.set_title("Vergleich", y=.86, pad=0, fontsize=15)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=6, azim=350)
    ax.tick_params(axis="both", labelsize=12)
    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)

    # ==========
    # show plots
    # ==========
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()




def compare_two_sequences_var(GT: str, ExGen: str, ExGen2: str):

    data = {}
    data2 = {}
    data3 = {}
    GT_data = pd.read_csv(GT)
    ExGen_data = pd.read_csv(ExGen)
    ExGen_data2 = pd.read_csv(ExGen2)
    duration = GT_data.iloc[-1, 0]

    # ====
    # Data
    # ====
    data[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data[i] = GT_data.iloc[:duration, i]

    array = [data[i].values for i in range(0, 16)]

    data2[0] = data[0]
    # data2[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data2[i] = ExGen_data.iloc[:duration, i]

    array2 = [data2[i].values for i in range(0, 16)]

    data3[0] = data[0]
    # data2[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data3[i] = ExGen_data2.iloc[:duration, i]

    array3 = [data3[i].values for i in range(0, 16)]

    # gs = GridSpec(1, 2, figure=fig)

    # ===========================
    # first subplot: Ground Truth
    # ===========================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1, zdir="y", color=colors[i - 1])

    # ax.set_title("Im Datensatz", y=.84, pad=0, fontsize=15)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)

    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # ====================================
    # second subplot: Expression Generator
    # ====================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, 16):
        ax.plot(array2[0], array2[i], i - 1, zdir="y", color=colors[i - 1])

    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)


    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # =======================================
    # third subplot: Expression Generator var
    # =======================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(1, 16):
        ax.plot(array3[0], array3[i], i - 1, zdir="y", color=colors[i - 1])

    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)


    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()


    # ====================================
    # fourth plot: 2 in 1 org
    # ====================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    width = 0.0
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1 + width, zdir="y", color=colors[i - 1], linestyle="--")
        ax.plot(array2[0], array2[i], i - 1, zdir="y", color=colors[i - 1])
        ax.add_collection3d(plt.fill_between(array[0], array[i], array2[i], color=colors[i-1], alpha=0.1), zs=i-1, zdir='y')

    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=6, azim=350)
    ax.tick_params(axis="both", labelsize=12)
    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # ====================================
    # fifth plot: 2 in 1 var
    # ====================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    width = 0.0
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1 + width, zdir="y", color=colors[i - 1], linestyle="--")
        ax.plot(array3[0], array3[i], i - 1, zdir="y", color=colors[i - 1])
        ax.add_collection3d(plt.fill_between(array[0], array[i], array3[i], color=colors[i - 1], alpha=0.1), zs=i - 1,
                            zdir='y')

    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(0, 15)
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=6, azim=350)
    ax.tick_params(axis="both", labelsize=12)
    fig.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, hspace=0, wspace=0)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()


###############
# TESTING STUFF
###############
def eval_one_sequence(sequence: str):
    # Data
    data = {}
    df = pd.read_csv(sequence)
    duration = df.iloc[-1, 0]
    for i in range(16):
        data[i] = df.iloc[:duration, i]
    array = [data[i].values for i in range(0, 16)]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
    for i in range(1, 16):
        ax2.plot(array[0], array[i], i - 1, zdir="y", color=colors[i - 1])
    # Stuff
    # ax2.set_title("GenEx: disgust to happy", y=.9, pad=0, fontsize=12)
    ax2.set_xlabel("Frame", fontsize=16)
    ax2.set_xlim(duration, 0)
    ax2.set_ylim(min(y_indexes), max(y_indexes))
    ax2.set_yticks(ticks=y_indexes)
    ax2.set_yticklabels(AUs, rotation=270)
    for ytick, color in zip(ax2.get_yticklabels(), colors):
        ytick.set_color(color)
    ax2.set_zlabel("AU-Intensity", fontsize=16)
    ax2.set_zlim(0, 1.25)
    ax2.view_init(elev=4, azim=340)
    ax2.tick_params(axis="both", labelsize=13)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig2.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.2, wspace=0.2)
    # plt.show()


##################
# 2D Loss Function
##################
def plot_loss(path: str):
    df = pd.read_csv(path, delimiter="\t")
    # print(df.head(7))
    MSE_train = df.iloc[1:, 7]/205
    MSE_val = df.iloc[1:, 4]/25
    MSE_test = 0.0340619147173129/25
    # print(MSE_train)
    # print(MSE_val)
    # print(MSE_test)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(MSE_train, label="MSE_train")
    ax.plot(MSE_val, label="MSE_val")

    ax.plot(100, 0.4654115445446223/25, ".", color="red", linewidth=0.5)
    ax.annotate(text=f"{0.4654115445446223/25:.4f}", color="red", xy=(100, 0.4654115445446223/25), xytext=(93, 0.4654115445446223/25+0.002), fontsize=14)

    ax.plot(200, 0.2358173221447418/25, ".", color="red", linewidth=0.5)
    ax.annotate(text=f"{0.2358173221447418/25:.4f}", color="red", xy=(200, 0.2358173221447418/25), xytext=(193, 0.2358173221447418/25+0.002), fontsize=14)

    ax.plot(355, MSE_test, ".", color="red", label="MSE_test",  linewidth=0.5)
    ax.annotate(text=f"{MSE_test:.4f}", color="red", xy=(355, MSE_test), xytext=(348, MSE_test+0.002), fontsize=14)

    ax.set_xlim(0, 360)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylabel("Error", fontsize=16)
    ax.set_ylim(0.00, 0.1)
    ax.set_yticks(np.arange(0.01, 0.1, 0.01))
    # ax.set_title("Durchschnittlicher MSE-Error pro Sequenz", fontsize=25)
    ax.legend(fontsize=15)
    ax.grid(True)
    # plt.show()


def plot_differences(GT: str, ExGen: str, ExGen2: str):

    data = {}
    data2 = {}
    GT_data = pd.read_csv(GT)
    ExGen_data = pd.read_csv(ExGen)
    ExGen_data2 = pd.read_csv(ExGen2)
    duration = GT_data.iloc[-1, 0]

    # ###########
    # First Plot
    data[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data[i] = GT_data.iloc[:duration, i] - ExGen_data.iloc[:duration, i]
    array = [data[i].values for i in range(0, 16)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for i in range(1, 16):
        ax.plot(array[0], array[i], i - 1, zdir="y", color=colors[i - 1])
    # Stuff
    # ax.set_title("Difference GT-ExGen: disgust to happy", y=.9, pad=0, fontsize=12)
    ax.set_xlabel("Frame", fontsize=16, labelpad=10)
    ax.set_xlim(duration, 0)
    # ax.set_ylim()
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=290)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=16, labelpad=10)
    ax.set_zticks(np.arange(-1.25, 1.25, 0.25))
    ax.set_zlim(-1.25, 1.25)
    ax.view_init(elev=3, azim=355)
    ax.tick_params(axis="both", labelsize=13)
    # ax.set_title("Difference GT - ExGen: Animation in DS", y=.83, pad=0, fontsize=15)

    # fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.2, wspace=0.2)
    fig.subplots_adjust(top=1.,
                        bottom=0.0,
                        left=0.00,
                        right=1.,
                        hspace=0.0,
                        wspace=0.0)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    # ###########
    # Second Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    data2[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data2[i] = GT_data.iloc[:duration, i] - ExGen_data2.iloc[:duration, i]
    array2 = [data2[i].values for i in range(0, 16)]

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for i in range(1, 16):
        ax.plot(array2[0], array2[i], i - 1, zdir="y", color=colors[i - 1])
    # Stuff
    # ax.set_title("Difference GT-ExGen: disgust to happy", y=.9, pad=0, fontsize=12)
    ax.set_xlabel("Frame", fontsize=16, labelpad=10)
    ax.set_xlim(duration, 0)
    # ax.set_ylim()
    ax.set_yticks(ticks=y_indexes)
    ax.set_yticklabels(AUs, rotation=290)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=16, labelpad=10)
    ax.set_zticks(np.arange(-1.25, 1.25, 0.25))
    ax.set_zlim(-1.25, 1.25)
    ax.view_init(elev=3, azim=355)
    ax.tick_params(axis="both", labelsize=13)
    # ax.set_title("Difference GT - ExGen: Animation not in DS", y=.83, pad=0, fontsize=15)

    fig.subplots_adjust(top=1.,
                        bottom=0.0,
                        left=0.00,
                        right=1.,
                        hspace=0.0,
                        wspace=0.0)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.show()


def plot_differences_box(GT: str, ExGen: str, ExGen2: str):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    data = {}
    data2 = {}
    GT_data = pd.read_csv(GT)
    ExGen_data = pd.read_csv(ExGen)
    ExGen_data2 = pd.read_csv(ExGen2)
    duration = GT_data.iloc[-1, 0]

    data[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data[i] = GT_data.iloc[:duration, i] - ExGen_data.iloc[:duration, i]
    array = [data[i].values for i in range(0, 16)]

    data2[0] = GT_data.iloc[:duration, 0]
    for i in range(1, 16):
        data2[i] = GT_data.iloc[:duration, i] - ExGen_data2.iloc[:duration, i]
    array2 = [data2[i].values for i in range(0, 16)]

    def set_box_color(bp, color, median):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=median, linestyle="dotted", linewidth=2.0)


    bpl = plt.boxplot(array[1:], positions=np.array(range(len(array[1:]))) * 2.0 - 0.39, sym='o', widths=0.72)
    set_box_color(bpl, '#5e84ff', "#000000")
    bpr = plt.boxplot(array2[1:], positions=np.array(range(len(array2[1:]))) * 2.0 + 0.39, sym='+', widths=0.72)
    set_box_color(bpr, '#ff4f4f', "#000000")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#2747b0', label='Animation im Datensatz')
    plt.plot([], c='#a62d2d', label='Animation nicht im Datensatz')
    plt.plot([], linestyle="dotted", c='#000000', label='Median')
    plt.legend(fontsize=15)


    plt.xticks(range(0, len(AUs) * 2, 2), AUs, fontsize=13)
    plt.xlim(-2, len(AUs) * 2)
    plt.ylim(-0.6, 0.6, 0.1)
    plt.yticks(np.arange(-0.5, 0.6, 0.1), fontsize=12)
    plt.grid(linestyle="dotted")
    # plt.savefig('boxcompare.png')
    plt.xlabel("Action Unit", fontsize=16)
    plt.ylabel("Abweichung zur Grundwahrheit", fontsize=16)

    plt.axhspan(-0.1, 0.1, facecolor='green', alpha=0.2)
    plt.axhspan(-0.2, -0.1, facecolor='#ffd059', alpha=0.2)
    plt.axhspan(0.2, 0.1, facecolor='#ffd059', alpha=0.2)
    plt.text(-1.8, 0.08, r"Abweichung nicht wahrnehmbar", fontsize=9, color="green", alpha=0.6)
    plt.text(-1.8, 0.18, r"Abweichung kaum wahrnehmbar", fontsize=9, color="#d99c00", alpha=1)

    # plt.show()


# ==================================== #
# =============== DATA =============== #
# ==================================== #
trainingshistory = "./history/history.txt"

# Groundtruth (always in DS ofc)
GT_disgusthappy5 = "Data/FaceTracker/preprocessed/csv/disgusthappy5_fill.csv"
GT_frownneutral5 = "Data/FaceTracker/preprocessed/csv/frownneutral5_fill.csv"

# Modell a)


# Modell b)


# Modell c)


# Modell d)


# Modell e)


# Modell f)


# Modell g)


# Modell h)


# Modell i)
i_disgusthappy5 = "./Data/Evaluation/i_testing/ExGen_i_testing_disgust2happy5_org.csv"
i_disgusthappy_v = "./Data/Evaluation/i_testing/ExGen_i_testing_disgust2happy_var.csv"

i_frownneutral5 = "./Data/Evaluation/i_testing/ExGen_i_testing_frown2neutral5_org.csv"
i_frownneutral_v = "./Data/Evaluation/i_testing/ExGen_i_testing_frown2neutral_var_bad_indataset.csv"

# ==================================== #
# ============ Functions ============= #
# ==================================== #

# Loss
# plot_loss(trainingshistory)

# ### Sequences ###
# eval_one_sequence(i_disgusthappy5)
# compare_two_sequences(GT_frownneutral5, i_frownneutral_v)
# compare_two_sequences_var(GT_frownneutral5, i_frownneutral5, i_frownneutral_v)

# ### Plot of difference ###
# plot_differences(GT_frownneutral5, i_frownneutral5, i_frownneutral_v)
plot_differences_box(GT_frownneutral5, i_frownneutral5, i_frownneutral_v)
plt.show()
