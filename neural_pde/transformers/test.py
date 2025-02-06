"""
Good deep learning practice - just make sure your model can fit a small dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from neural_pde.transformers.base_transformer import Transformer, TransformerConfig

# regression with transformer

config = TransformerConfig(input_dim=1, output_dim=1, n_embd=256, n_head=4, n_layer=6, bias=True, dropout=0.1)
model = Transformer(config)

x = torch.linspace(0, 10, 1000)
y = 3.0 * torch.sin(x) + 2.0

plt.plot(x, y)
plt.show()

# Training

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
losses = []
epoch_idx = []
for i in range(1000):
    random_idx = torch.randint(0, len(x), (32,))
    x_train = x[random_idx]
    y_train = y[random_idx]

    x_train = x_train.unsqueeze(1)
    y_train = y_train.unsqueeze(1)

    y_pred = model.no_seq_forward(x_train)
    loss = loss_fn(y_pred, y_train)
    if i % 30 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}")
        losses.append(loss.item())
        epoch_idx.append(i)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
y_hat = model.no_seq_forward(x.unsqueeze(1))

plt.plot(x, y_hat.detach().numpy(), label="pred", color="red")
plt.plot(x, y, label="true", color="blue")
plt.legend()
plt.show()

plt.plot(epoch_idx, losses)
plt.title("Loss vs Epoch")
plt.show()
