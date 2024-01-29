import math
import matplotlib.pyplot as plt
from IPython import display
from time import sleep

def flipbook(out, s=0):
    for o in out:
        display.clear_output(wait=True)
        display.display(o)
        sleep(s)

def grid(out, show=5, range=None):
    if(range is None):
        range = (0, len(out)-1)
    fig, ax = plt.subplots(1,show)
    fig.set_figwidth(20)
    for i, x in enumerate(ax):
        oidx = int((range[1]-range[0])*i/(show-1))+range[0]
        x.set_title(oidx)
        x.imshow(out[oidx])

def single(out):
    display.display(out)

