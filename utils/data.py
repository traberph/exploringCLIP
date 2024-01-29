import numpy as np


locations7 = ['', 'on the moon', 'in the jungle', 'at the beach', 'in the city', 'on a mountain', 'in the snow', 'in the woods']
locations3 = ['', 'on the moon', 'in the jungle', 'at the beach']
colors3 = ['blue', 'orange', 'red']
colors8 = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'white', 'purple']
objects3 = ['monkey', 'child', 'car']
sentences = [[f'a {color} {object}', color, object] for color in colors3 for object in objects3]


descriptions = np.array(sentences)
