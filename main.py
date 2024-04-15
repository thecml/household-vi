import numpy as np
import torch
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
from torch import distributions
from model import Model

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return x[:, None], y[:, None]

if __name__ == "__main__":
    X, y = load_dataset()
    
    epochs = 2000

    def det_loss(y, y_pred, model):
        batch_size = y.shape[0]
        reconstruction_error = -distributions.Normal(y_pred, .1).log_prob(y).sum()
        kl = model.accumulated_kl_div
        model.reset_kl_div()
        return reconstruction_error + kl

    m = Model(1, 20, 1, n_batches=1)
    optim = torch.optim.Adam(m.parameters(), lr=0.01)

    for epoch in range(epochs):
        optim.zero_grad()
        y_pred = m(X)
        loss = det_loss(y_pred, y, m)
        loss.backward()
        optim.step()
    
    with torch.no_grad():
        trace = np.array([m(X).flatten().numpy() for _ in range(1000)]).T
    q_25, q_95 = np.quantile(trace, [0.05, 0.95], axis=1)
    plt.figure(figsize=(16, 6))
    plt.plot(X, trace.mean(1))
    plt.scatter(X, y)
    plt.fill_between(X.flatten(), q_25, q_95, alpha=0.2)