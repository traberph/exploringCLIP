from utils.model import CLIP
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def matrix(prompts):
    # encode prompt
    clip = CLIP()
    embeddings = clip.embed(prompts, pooled=True)
    
    vs = [(x-y).abs().sort(descending=True) for x in embeddings for y in embeddings]

    # visualize
    plt.rcParams['figure.figsize'] = [15, 15]
    fig, ax = plt.subplots(len(embeddings), len(embeddings))
    for idx, res in enumerate(vs):
        x = idx%len(embeddings)
        y = int(idx/len(embeddings))
        if x==y:
            # diagonal
            ax[x][y].set_axis_off()
            ax[x][y].text(0.1,0.5 ,prompts[x])
        elif x>y:
            # lower half
            sns.lineplot(res[0].cpu(), ax=ax[x][y])
        else:
            # upper half
            bars = 10
            p = res[0][:bars]#p[0:50]
            sns.barplot(p.cpu(), ax=ax[x][y])
            label = list(map(lambda x:x.item(), res[1][:bars]))
            ax[x][y].bar_label(ax[x][y].containers[0], labels=label, label_type='center', rotation=90, color='white')


def summarize(prompts):
    # encode prompts
    clip = CLIP()
    embeddings = clip.embed(prompts, pooled=True)

    # calculate differences in dimensions and square them
    vs = [(x-y)**2 for x in embeddings for y in embeddings]

    # sum the differences 
    sum = torch.zeros(768)
    for t in vs:
        sum += t.cpu() 
    
    # visualize
    bars = 20
    s = sum.sort(descending=True)
    plt.rcParams['figure.figsize'] = [15, 5]
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2]})
    sns.lineplot(s.values, ax=ax[0])
    sns.barplot(s.values[:bars], ax=ax[1])
    labels = list(map(lambda x:x.item(), s.indices[:bars]))
    ax[1].bar_label(ax[1].containers[0], labels=labels, label_type='center', rotation=90, color='white')