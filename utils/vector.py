from tqdm.notebook import tqdm_notebook as tqdm
import torch

# interploate multidemensional vectors x,y in steps (slow cpu)
def interp(x, y, steps):
    res = np.empty((steps,len(x)))
    for i in tqdm(range(len(x))):
        step = (y[i]-x[i])/(steps+1)
        for s in range(steps):
            res[s][i]= (x[i]+step*(s+1))
    return res

# interploate much faster (fast gpu)
def interp_gpu(x, y, steps):
    # Expand dimensions to match the target shape
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)

    # Create a linear space tensor for steps
    s_values = torch.linspace(0, 1, steps, device='cuda')

    # Expand dimensions to allow broadcasting
    s_values = s_values.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Calculate the entire interpolation in a single operation
    res = x + (y - x) * s_values

    return res.squeeze(0).squeeze(0)

# generates always the same tensor and returns it batch_size times
def getNoise(batch_size):
    t = torch.randn((4,64,64), generator=torch.manual_seed(0))
    return t.repeat((batch_size, 1,1,1))