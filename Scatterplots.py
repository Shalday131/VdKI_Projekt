import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_excel("Features 3.xlsx")

sns.set(style="darkgrid")

# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Anzahl Ecken", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Ecken über Anzahl Kreise")
# fig.savefig("1 Anzahl Ecken über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Anzahl Linien", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Linien über Anzahl Kreise")
# fig.savefig("2 Anzahl Linien über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Anzahl Orbs", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Orbs über Anzahl Kreise")
# fig.savefig("3 Anzahl Orbs über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Anzahl Keypoints", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Keypoints über Anzahl Kreise")
# fig.savefig("4 Anzahl Keypoints über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Anzahl Kreise")
# fig.savefig("5 Aspect Ratio über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Kreise", y="Maximaler Histogrammwert", hue="Label", height=5, aspect=2, legend_out=False).set(title="Maximaler Histogrammwert über Anzahl Kreise")
# fig.savefig("6 Maximaler Histogrammwert über Anzahl Kreise.png")
#
# fig = sns.catplot(data=df, x="Anzahl Ecken", y="Anzahl Linien", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Linien über Anzahl Ecken")
# fig.savefig("7 Anzahl Linien über Anzahl Ecken.png")
#
# fig = sns.catplot(data=df, x="Anzahl Ecken", y="Anzahl Orbs", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Orbs über Anzahl Ecken")
# fig.savefig("8 Anzahl Orbs über Anzahl Ecken.png")
#
# fig = sns.catplot(data=df, x="Anzahl Ecken", y="Anzahl Keypoints", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Keypoints über Anzahl Ecken")
# fig.savefig("9 Anzahl Keypoints über Anzahl Ecken.png")
#
# fig = sns.catplot(data=df, x="Anzahl Ecken", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Anzahl Ecken")
# fig.savefig("10 Aspect Ratio über Anzahl Ecken.png")
#
# fig = sns.catplot(data=df, x="Anzahl Ecken", y="Maximaler Histogrammwert", hue="Label", height=5, aspect=2, legend_out=False).set(title="Maximaler Histogrammwert über Anzahl Ecken")
# fig.savefig("11 Maximaler Histogrammwert über Anzahl Ecken.png")
#
# fig = sns.catplot(data=df, x="Anzahl Linien", y="Anzahl Orbs", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Orbs über Anzahl Linien")
# fig.savefig("12 Anzahl Orbs über Anzahl Linien.png")
#
# fig = sns.catplot(data=df, x="Anzahl Linien", y="Anzahl Keypoints", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Keypoints über Anzahl Linien")
# fig.savefig("13 Anzahl Keypoints über Anzahl Linien.png")
#
# fig = sns.catplot(data=df, x="Anzahl Linien", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Anzahl Linien")
# fig.savefig("14 Aspect Ratio über Anzahl Linien.png")
#
# fig = sns.catplot(data=df, x="Anzahl Linien", y="Maximaler Histogrammwert", hue="Label", height=5, aspect=2, legend_out=False).set(title="Maximaler Histogrammwert über Anzahl Linien")
# fig.savefig("15 Maximaler Histogrammwert über Anzahl Linien.png")
#
# fig = sns.catplot(data=df, x="Anzahl Orbs", y="Anzahl Keypoints", hue="Label", height=5, aspect=2, legend_out=False).set(title="Anzahl Keypoints über Anzahl Orbs")
# counter = 0
# for ax in fig.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     for i,l in enumerate(labels):
#         counter = counter+1
#         if(counter%50 != 0): labels[i] = '' # skip even labels
#     ax.set_xticklabels(labels) # set new labels
# fig.savefig("16 Anzahl Keypoints über Anzahl Orbs.png")
#
# fig = sns.catplot(data=df, x="Anzahl Orbs", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Anzahl Orbs")
# counter = 0
# for ax in fig.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     for i,l in enumerate(labels):
#         counter = counter+1
#         if(counter%50 != 0): labels[i] = '' # skip even labels
#     ax.set_xticklabels(labels) # set new labels
# fig.savefig("17 Aspect Ratio über Anzahl Orbs.png")
#
# fig = sns.catplot(data=df, x="Anzahl Orbs", y="Maximaler Histogrammwert", hue="Label", height=5, aspect=2, legend_out=False).set(title="Maximaler Histogrammwert über Anzahl Orbs")
# counter = 0
# for ax in fig.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     for i,l in enumerate(labels):
#         counter = counter+1
#         if(counter%50 != 0): labels[i] = '' # skip even labels
#     ax.set_xticklabels(labels) # set new labels
# fig.savefig("18 Maximaler Histogrammwert über Anzahl Orbs.png")
#
# fig = sns.catplot(data=df, x="Anzahl Keypoints", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Anzahl Keypoints")
# counter = 0
# for ax in fig.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     for i,l in enumerate(labels):
#         counter = counter+1
#         if(counter%50 != 0): labels[i] = '' # skip even labels
#     ax.set_xticklabels(labels) # set new labels
# fig.savefig("19 Aspect Ratio über Anzahl Keypoints.png")
#
# fig = sns.catplot(data=df, x="Anzahl Keypoints", y="Maximaler Histogrammwert", hue="Label", height=5, aspect=2, legend_out=False).set(title="Maximaler Histogrammwert über Anzahl Keypoints")
# counter = 0
# for ax in fig.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     for i,l in enumerate(labels):
#         counter = counter+1
#         if(counter%50 != 0): labels[i] = '' # skip even labels
#     ax.set_xticklabels(labels) # set new labels
# fig.savefig("20 Maximaler Histogrammwert über Anzahl Keypoints.png")

fig = sns.catplot(data=df, x="Maximaler Histogrammwert", y="Aspect Ratio", hue="Label", height=5, aspect=2, legend_out=False).set(title="Aspect Ratio über Maximaler Histogrammwert")
counter = 0
for ax in fig.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    for i,l in enumerate(labels):
        counter = counter+1
        if(counter%50 != 0): labels[i] = '' # skip even labels
    ax.set_xticklabels(labels) # set new labels
fig.savefig("21 Aspect Ratio über Maximaler Histogrammwert.png")
"""
del df[df.columns[0]]
fig23 = sns.pairplot(df, hue="Label")
fig23.savefig("Scattermatrix.png")
"""

