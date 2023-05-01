import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as progress_bar
from torch.utils.data import DataLoader
import numpy as np


class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, penalty):
        super(LocalDictionaryLoss, self).__init__()
        self.penalty = penalty

    def forward(self, A, y, x):
        return self.forward_detailed(A, y, x)[2]

    def forward_detailed(self, A, y, x):
        weight = (y.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=2)
        a = 0.5 * (y - x @ A).pow(2).sum(dim=1).mean()
        b = (weight * x).sum(dim=1).mean()
        return a, b, a + b * self.penalty


class KDS(nn.Module):
    def __init__(
            self,
            num_layers,
            input_size,
            hidden_size,
            penalty,
            accelerate=True,
            train_step=True,
            W=None,
            step=None,
    ):
        super(KDS, self).__init__()

        # hyperparameters
        self.register_buffer("num_layers", torch.tensor(int(num_layers)))
        self.register_buffer("input_size", torch.tensor(int(input_size)))
        self.register_buffer("hidden_size", torch.tensor(int(hidden_size)))
        self.register_buffer("penalty", torch.tensor(float(penalty)))
        self.register_buffer("accelerate", torch.tensor(bool(accelerate)))

        # parameters
        if W is None:
            # W = torch.zeros(self.hidden_size, self.input_size)
            W = torch.randn(self.hidden_size, self.input_size)
        self.register_parameter("W", torch.nn.Parameter(W))
        if step is None:
            step = W.svd().S[0] ** -2
        if train_step:
            self.register_parameter("step", torch.nn.Parameter(step))
        else:
            self.register_buffer("step", step)

    def get_penalty(self):
        return self.penalty

    def forward(self, y):
        x = self.encode(y)
        y = self.decode(x)
        return y

    def encode(self, y):
        if self.accelerate:
            return self.encode_accelerated(y)
        else:
            return self.encode_basic(y)

    def encode_basic(self, y):
        x = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        # weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
        weight = (
                y.square().sum(dim=1, keepdims=True)
                + self.W.T.square().sum(dim=0, keepdims=True)
                - 2 * y @ self.W.T
        )
        for layer in range(self.num_layers):
            grad = (x @ self.W - y) @ self.W.T
            grad = grad + weight * self.penalty
            x = self.activate(x - grad * self.step)
        return x

    def encode_accelerated(self, y):
        x_tmp = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        x_old = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        # weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
        weight = (y.square().sum(dim=1, keepdims=True) + self.W.T.square().sum(dim=0, keepdims=True)- 2 * y @ self.W.T)
        for layer in range(self.num_layers):
            grad = (x_tmp @ self.W - y) @ self.W.T
            grad = grad + weight * self.penalty
            x_new = self.activate(x_tmp - grad * self.step)
            x_old, x_tmp = x_new, x_new + layer / (layer + 3) * (x_new - x_old)
        return x_new

    def decode(self, x):
        return x @ self.W

    def activate(self, x):
        m, n = x.shape
        cnt_m = torch.arange(m, device=x.device)
        cnt_n = torch.arange(n, device=x.device)
        u = x.sort(dim=1, descending=True).values
        v = (u.cumsum(dim=1) - 1) / (cnt_n + 1)
        w = v[cnt_m, (u > v).sum(dim=1) - 1]
        return (x - w.view(m, 1)).relu()


def run_kds_experiment(lam, data, net, lr, epochs, batch_size, device):

    with torch.no_grad():
        p = torch.randperm(len(data))[: net.hidden_size]
        net.W.data = data[p]
        net.step.fill_((net.W.data.svd()[1][0] ** -2).item())

    #device = 'cuda'
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = LocalDictionaryLoss(torch.tensor(float(lam)))

    net.train()

    kds_loss = []
    sparsity_of_coefs = []

    dataloader = DataLoader(data, batch_size=batch_size)

    for epoch in progress_bar(range(epochs)):

        epoch_loss = []
        sparsity_levels = []

        shuffle = torch.randperm(len(data))

        data = data[shuffle]

        for img_batch in dataloader:
            y = img_batch.to(device)
            x_hat = net.encode(y)
            loss = criterion(net.W, y, x_hat)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimizer.step()
            # compute loss for each batch
            # store in accumulated value

            avg_sparsity_coef = torch.mean(torch.count_nonzero(x_hat, dim=1).to(torch.float64))
            epoch_loss.append(loss.item())
            sparsity_levels.append(avg_sparsity_coef.item())

        kds_loss.append(np.mean(epoch_loss))
        sparsity_of_coefs.append(np.mean(sparsity_levels))

    with torch.no_grad():
        net.eval()
        x_hat = []
        for img_batch in dataloader:
            y = img_batch.to(device)
            x_hat.append(net.encode(y).cpu())
        x_hat = torch.cat(x_hat)

    # save x_hat and net.W
    #torch.save(net.W, 'dic_lambda_' + str(lam) + '.pth')
    #torch.save(x_hat, 'representation_lambda_' + str(lam) + '.pth')

    # torch.load('model_lambda_0.pth')

    return kds_loss, net.W.cpu().detach().numpy(), x_hat.cpu().detach().numpy()
