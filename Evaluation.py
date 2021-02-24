from sys import path
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


AUs = ["AU1L", "AU1R", "AU2L", "AU2R", "AU4L", "AU4R", "AU6L", "AU6R", "AU9", "AU10", "AU13L", "AU13R", "AU18",
       "AU22", "AU27"]
AUs_reduced = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU10", "AU13", "AU18", "AU22", "AU27"]
colors = ["#ff4a47", "#ff4a47", "#fa8b2a", "#fa8b2a", "#d1ce00", "#d1ce00", "#6aff00", "#6aff00", "#00eeff", "gray",
          "#a600ff", "#a600ff", "black", "#007bff", "red"]
colors_reduced = ["#ff4a47", "#fa8b2a", "#d1ce00", "#6aff00", "#00eeff", "gray", "#a600ff", "black", "#007bff", "red"]
y_indexes = np.arange(len(AUs))
y_indexes_reduced = np.arange(len(AUs_reduced))


def compare_two_sequences(GT_sequence: str, ExGen_sequence: str):
    fig = plt.figure()
    # ===========================
    # first subplot: Ground Truth
    # ===========================
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # Data
    data = {}
    df = pd.read_csv(GT_sequence)
    print(df.columns)
    duration = df.iloc[-1, 0]
    for i in range(16):
        data[i] = df.iloc[:duration, i]
    Frame = data[0].values
    AU1L = data[1].values
    AU1R = data[2].values
    AU2L = data[3].values
    AU2R = data[4].values
    AU4L = data[5].values
    AU4R = data[6].values
    AU6L = data[7].values
    AU6R = data[8].values
    AU9 = data[9].values
    AU10 = data[10].values
    AU13L = data[11].values
    AU13R = data[12].values
    AU18 = data[13].values
    AU22 = data[14].values
    AU27 = data[15].values
    # Plot
    ax.plot(Frame, AU1L, 0, zdir="y", color="#ff4a47")
    ax.plot(Frame, AU1R, 1, zdir="y", color="#ff4a47")
    ax.plot(Frame, AU2L, 3, zdir="y", color="#fa8b2a")
    ax.plot(Frame, AU2R, 2, zdir="y", color="#fa8b2a")
    ax.plot(Frame, AU4L, 4, zdir="y", color="#d1ce00")
    ax.plot(Frame, AU4R, 5, zdir="y", color="#d1ce00")
    ax.plot(Frame, AU6L, 6, zdir="y", color="#6aff00")
    ax.plot(Frame, AU6R, 7, zdir="y", color="#6aff00")
    ax.plot(Frame, AU9, 8, zdir="y", color="#00eeff")
    ax.plot(Frame, AU10, 9, zdir="y", color="gray")
    ax.plot(Frame, AU13L, 10, zdir="y", color="#a600ff")
    ax.plot(Frame, AU13R, 11, zdir="y", color="#a600ff")
    ax.plot(Frame, AU18, 12, zdir="y", color="black")
    ax.plot(Frame, AU22, 13, zdir="y", color="#007bff")
    ax.plot(Frame, AU27, 14, zdir="y", color="red")
    # Stuff

    ax.set_title("GT: disgust to happy", y=.9, pad=0, fontsize=15)
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

    # ====================================
    # second subplot: Expression Generator
    # ====================================
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    # Data
    data = {}
    df = pd.read_csv(ExGen_sequence)
    print(df.columns)
    for i in range(16):
        data[i] = df.iloc[:duration, i]
    # short:
    # array = [data[i].values for i in range(0, 15)]
    Frame = df.iloc[:duration, 0]
    AU1L = data[1].values
    AU1R = data[2].values
    AU2L = data[3].values
    AU2R = data[4].values
    AU4L = data[5].values
    AU4R = data[6].values
    AU6L = data[7].values
    AU6R = data[8].values
    AU9 = data[9].values
    AU10 = data[10].values
    AU13L = data[11].values
    AU13R = data[12].values
    AU18 = data[13].values
    AU22 = data[14].values
    AU27 = data[15].values
    # # Plot
    # ax.plot(Frame, AU1L, 0, zdir="y", color="#ff4a47")
    # ax.plot(Frame, AU1R, 1, zdir="y", color="#ff4a47")
    # ax.plot(Frame, AU2L, 3, zdir="y", color="#fa8b2a")
    # ax.plot(Frame, AU2R, 2, zdir="y", color="#fa8b2a")
    # ax.plot(Frame, AU4L, 4, zdir="y", color="#d1ce00")
    # ax.plot(Frame, AU4R, 5, zdir="y", color="#d1ce00")
    # ax.plot(Frame, AU6L, 6, zdir="y", color="#6aff00")
    # ax.plot(Frame, AU6R, 7, zdir="y", color="#6aff00")
    # ax.plot(Frame, AU9, 8, zdir="y", color="#00eeff")
    # ax.plot(Frame, AU10, 9, zdir="y", color="gray")
    # ax.plot(Frame, AU13L, 10, zdir="y", color="#a600ff")
    # ax.plot(Frame, AU13R, 11, zdir="y", color="#a600ff")
    # ax.plot(Frame, AU18, 12, zdir="y", color="black")
    # ax.plot(Frame, AU22, 13, zdir="y", color="#007bff")
    # ax.plot(Frame, AU27, 14, zdir="y", color="red")
    #
    # # Stuff
    # ax.set_title("GenEx: disgust to happy", y=.9, pad=0)
    # ax.set_xlabel("Frame")
    # ax.set_xlim(duration, 0)
    # ax.set_ylim(0, 15)
    # ax.set_yticks(ticks=y_indexes)
    # ax.set_yticklabels(AUs, rotation=270)
    # for ytick, color in zip(ax.get_yticklabels(), colors):
    #     ytick.set_color(color)
    # ax.set_zlabel("AU-Intensity")
    # ax.set_zlim(0, 1.25)
    # Plot
    width = 0.00
    ax.plot(Frame, AU1L, 0 - width, zdir="y", color="#ff4a47", linestyle="--")
    ax.plot(Frame, AU1R, 0 + width, zdir="y", color="#ff4a47")
    ax.plot(Frame, AU2L, 1 - width, zdir="y", color="#fa8b2a", linestyle="--")
    ax.plot(Frame, AU2R, 1 + width, zdir="y", color="#fa8b2a")
    ax.plot(Frame, AU4L, 2 - width, zdir="y", color="#d1ce00", linestyle="--")
    ax.plot(Frame, AU4R, 2 + width, zdir="y", color="#d1ce00")
    ax.plot(Frame, AU6L, 3 - width, zdir="y", color="#6aff00", linestyle="--")
    ax.plot(Frame, AU6R, 3 + width, zdir="y", color="#6aff00")
    ax.plot(Frame, AU9, 4, zdir="y", color="#00eeff")
    ax.plot(Frame, AU10, 5, zdir="y", color="gray")
    ax.plot(Frame, AU13L, 6 - width, zdir="y", color="#a600ff", linestyle="--")
    ax.plot(Frame, AU13R, 6 + width, zdir="y", color="#a600ff")
    ax.plot(Frame, AU18, 7, zdir="y", color="black")
    ax.plot(Frame, AU22, 8, zdir="y", color="#007bff")
    ax.plot(Frame, AU27, 9, zdir="y", color="red")
    # Stuff
    ax.set_title("GenEx: disgust to happy", y=.9, pad=0, fontsize=15)
    ax.set_xlabel("Frame", fontsize=14)
    ax.set_xlim(duration, 0)
    ax.set_ylim(min(y_indexes_reduced), max(y_indexes_reduced))
    ax.set_yticks(ticks=y_indexes_reduced)
    ax.set_yticklabels(AUs_reduced, rotation=270)
    for ytick, color in zip(ax.get_yticklabels(), colors_reduced):
        ytick.set_color(color)
    ax.set_zlabel("AU-Intensity", fontsize=14)
    ax.set_zlim(0, 1.25)
    # color=("#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47")
    ax.view_init(elev=4, azim=340)
    ax.tick_params(axis="both", labelsize=12)
    # ==========
    # show plots
    # ==========
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.subplots_adjust(top=0.995, bottom=0.015, left=0.002, right=0.98, hspace=0.2, wspace=0.011)
    # plt.show()


########################################
# TESTING STUFF:
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


def plot_differences(GT: str, ExGen: str):
    # Data
    data = {}
    GT_data = pd.read_csv(GT)
    ExGen_data = pd.read_csv(ExGen)
    duration = GT_data.iloc[-1, 0]

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
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.2, wspace=0.2)

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

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    bpr = plt.boxplot(array2[1:], positions=np.array(range(len(array2[1:]))) * 2.0 + 0.4, sym='+', widths=0.6)
    set_box_color(bpr, '#2C7BB6')
    bpl = plt.boxplot(array[1:], positions=np.array(range(len(array[1:]))) * 2.0 - 0.4, sym='+', widths=0.6)
    set_box_color(bpl, '#D7191C')


    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='within Dataset')
    plt.plot([], c='#2C7BB6', label='Variation')
    plt.legend()

    plt.xticks(range(0, len(AUs) * 2, 2), AUs)
    plt.xlim(-2, len(AUs) * 2)
    plt.ylim(-1.25, 1.25, 0.25)
    plt.yticks(np.arange(-1.25, 1.25, 0.25))
    plt.tight_layout()
    plt.grid(True)
    # plt.savefig('boxcompare.png')

    # plt.show()


# ==================================== #
# =============== DATA =============== #
# ==================================== #
trainingshistory = "./history/history.txt"
GT_seq = "Data/FaceTracker/preprocessed/csv/disgusthappy1_fill.csv"
ExGen_seq_org = "./Data/Evaluation/i_testing/ExGen_i_testing_disgust2happy.csv"
ExGen_seq_var = "./Data/Evaluation/i_testing/ExGen_i_i_testing_disgust2happy_var.csv"




# Loss
# plot_loss(trainingshistory)


# # Plot of difference? Some AUs or all?
plot_differences(GT_seq, ExGen_seq_org)
plot_differences_box(GT_seq, ExGen_seq_org, ExGen_seq_var)
#
#
# # Sequences - which/how many?
# compare_two_sequences(GT_seq, ExGen_seq)
# eval_one_sequence(GT_seq)

plt.show()
