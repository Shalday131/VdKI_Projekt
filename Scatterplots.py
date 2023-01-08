import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("Features 3.xlsx")

sns.set(style="darkgrid")

fig1 = sns.catplot(data=df, x="Label", y="Anzahl Kreise")
fig1.savefig("out.png")

fig2 = sns.catplot(data=df, x="Anzahl Kreise", y="Anzahl Ecken", hue="Label")
fig2.savefig("out2.png")


