from sys import path
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()

AUs = ["AU1L", "AU1R", "AU2L", "AU2R", "AU4L", "AU4R", "AU6L", "AU6R", "AU9", "AU10", "AU13L", "AU13R", "AU18", "AU22", "AU27"]
y_indexes = np.arange(len(AUs))

# =============================
# first subplot: Ground Truth
# =============================
ax = fig.add_subplot(1, 2, 1, projection="3d")

# Data
actionunit = {}
data = pd.read_csv("Data/FaceTracker/preprocessed/csv/disgusthappy1_fill.csv")
duration = data.iloc[-1, 0]
for i in range(16):
    actionunit[i] = data.iloc[:duration, i]
Frame = actionunit[0].values
AU1L = actionunit[1].values
AU1R = actionunit[2].values
AU2L = actionunit[3].values
AU2R = actionunit[4].values
AU4L = actionunit[5].values
AU4R = actionunit[6].values
AU6L = actionunit[7].values
AU6R = actionunit[8].values
AU9 = actionunit[9].values
AU10 = actionunit[10].values
AU13R = actionunit[11].values
AU13L = actionunit[12].values
AU18 = actionunit[13].values
AU22 = actionunit[14].values
AU27 = actionunit[15].values


# Plot
ax.plot(Frame, AU1L, 0, zdir="y", color="#ff4a47")
ax.plot(Frame, AU1R, 1, zdir="y", color="#ff4a47")
ax.plot(Frame, AU2R, 2, zdir="y", color="#fa8b2a")
ax.plot(Frame, AU2L, 3, zdir="y", color="#fa8b2a")
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
colors = ["#ff4a47", "#ff4a47", "#fa8b2a", "#fa8b2a", "#d1ce00", "#d1ce00", "#6aff00", "#6aff00", "#00eeff", "gray", "#a600ff", "#a600ff", "black", "#007bff", "red"]
ax.set_title("GT: disgust to happy", y=.9, pad=0)
ax.set_xlabel("Frame")
ax.set_xlim(duration, 0)
ax.set_ylim(0, 15)
ax.set_yticks(ticks=y_indexes)
ax.set_yticklabels(AUs, rotation=270)
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)
ax.set_zlabel("AU-Intensity")
ax.set_zlim(0, 1.25)

ax.view_init(elev=4, azim=340)

# ======================================
# second subplot: Expression Generator
# ======================================
ax = fig.add_subplot(1, 2, 2, projection="3d")

# Data
actionunit = {}
data = pd.read_csv("./Data/Evaluation/i_testing/ExGen_i_testing_disgust2happy.csv")
for i in range(16):
    actionunit[i] = data.iloc[:duration, i]

Frame = data.iloc[:duration, 0]
AU1L = actionunit[1].values
AU1R = actionunit[2].values
AU2L = actionunit[3].values
AU2R = actionunit[4].values
AU4L = actionunit[5].values
AU4R = actionunit[6].values
AU6L = actionunit[7].values
AU6R = actionunit[8].values
AU9 = actionunit[9].values
AU10 = actionunit[10].values
AU13R = actionunit[11].values
AU13L = actionunit[12].values
AU18 = actionunit[13].values
AU22 = actionunit[14].values
AU27 = actionunit[15].values


# Plot
ax.plot(Frame, AU1L, 0, zdir="y", color="#ff4a47")
ax.plot(Frame, AU1R, 1, zdir="y", color="#ff4a47")
ax.plot(Frame, AU2R, 2, zdir="y", color="#fa8b2a")
ax.plot(Frame, AU2L, 3, zdir="y", color="#fa8b2a")
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



ax.set_title("GenEx: disgust to happy", y=.9, pad=0)
ax.set_xlabel("Frame")
ax.set_xlim(duration, 0)
ax.set_ylim(0, 15)
ax.set_yticks(ticks=y_indexes)
ax.set_yticklabels(AUs, rotation=270)
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)
ax.set_zlabel("AU-Intensity")
ax.set_zlim(0, 1.25)

# color=("#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47", "#ff4a47")

ax.view_init(elev=4, azim=340)


# ======================================
# show plots
# ======================================
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
fig.subplots_adjust(top=0.995, bottom=0.015, left=0.002, right=0.98, hspace=0.2, wspace=0.011)
plt.show()

