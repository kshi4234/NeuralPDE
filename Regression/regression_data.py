import torch

# Function is literally just y = 3x
def func(x):
    return 3 * x

# Batching not implemented
def gen_data(datapoints=100, batch_size=1, train_ratio=0.8, train=True):
    x = torch.linspace(0, 10, steps=datapoints).reshape(datapoints, 1)
    y = func(x).squeeze()
    
    if train:
        train_x, val_x = x[:int(train_ratio*datapoints)], x[int(train_ratio*datapoints):]
        train_y, val_y = y[:int(train_ratio*datapoints)], y[int(train_ratio*datapoints):]
        return train_x, train_y, val_x, val_y
    
    return x, y