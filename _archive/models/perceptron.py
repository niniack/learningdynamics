from copy import deepcopy

import torch
from torch import nn


class SingleNeuronModel(nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.features(x)

    def vectorize_weights(self, layer=0):
        all_weights = [self.features[0].weight.detach()]
        return torch.flatten(torch.cat(all_weights), start_dim=0, end_dim=-1)


class HiddenLayerModel(nn.Module):
    def __init__(self, neurons=1, in_features=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=neurons, bias=False),
            nn.Tanh(),
            nn.Linear(in_features=neurons, out_features=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.features(x)

    def vectorize_weights(self, layer=0):
        all_weights = [
            self.features[0].weight.detach(),
            # self.features[2].weight.detach(),
        ]
        for i, w in enumerate(all_weights):
            all_weights[i] = torch.flatten(w)
        return torch.cat(all_weights, dim=0)


class MLP(nn.Module):
    def __init__(self, config=[8], in_features=2, nonlinearity="tanh"):
        super().__init__()

        if nonlinearity == "tanh":
            act = nn.Tanh()
        elif nonlinearity == "relu":
            act = nn.ReLU()
        else:
            raise NotImplementedError()

        # Init container
        self.features = nn.Sequential()

        # First layer
        self.features.append(
            nn.Linear(in_features=in_features, out_features=config[0], bias=False)
        )
        self.features.append(deepcopy(act))

        for i, _ in enumerate(config[1:], start=1):
            self.features.append(
                nn.Linear(in_features=config[i - 1], out_features=config[i], bias=True)
            )
            self.features.append(deepcopy(act))

        # Last layer
        self.features.append(
            nn.Linear(in_features=config[-1], out_features=2, bias=False)
        )

    def forward(self, x):
        logits = self.features(x)
        return logits
        # return torch.softmax(logits, dim=-1)

    def vectorize_weights(self, layer=0):
        all_weights = [
            self.features[layer].weight.detach(),
        ]
        for i, w in enumerate(all_weights):
            all_weights[i] = torch.flatten(w)
        return torch.cat(all_weights, dim=0)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.weight, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
