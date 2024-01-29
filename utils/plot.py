import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot(data, meta=None):
    plt.rcParams['figure.figsize'] = [10, 10]
    if meta is None:
        p = sns.scatterplot(x=data[:,0], y=data[:,1])
    else:
        p = sns.scatterplot(x=data[:,0], y=data[:,1],hue=meta[:,1], style=meta[:,2], s=80)
        sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
    
    